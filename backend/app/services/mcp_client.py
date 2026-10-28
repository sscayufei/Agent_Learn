# 最小 stdio MCP 客户端：启动子进程 -> initialize 握手 -> JSON-RPC 调工具 -> 关闭进程。
# 请求携带 ID 标识，当前客户端按顺序读单个响应；stdout 属于协议通道，普通日志应写 stderr。
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings


class McpClientError(RuntimeError):
    pass


class StdioMcpClient:
    def __init__(self, command: str | None = None, args: list[str] | None = None, cwd: str | None = None) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.command = command or sys.executable
        self.args = args or [str(backend_root / "app" / "mcp" / "memory_server.py")]
        self.cwd = cwd or str(backend_root)
        self._next_id = 1

    # 每次调用启动一个 stdio 子进程，先 initialize，再通知 initialized，最后发 tools/call。
    # finally 无论成功失败都停止进程，避免工具异常后留下后台解释器。
    # 当前实现顺序等待单行响应，不是支持并发/任意通知分流的通用完整 MCP SDK。
    # 请求带递增 ID 便于协议标识，但不能把这段代码描述成已实现完整响应 ID 配对验证。
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        process = await self._start()
        try:
            await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            return await self._request(process, "tools/call", {"name": name, "arguments": arguments})
        finally:
            await self._stop(process)

    # 通过真实 initialize 握手和 tools/list 获取服务端能力，用于界面 MCP 状态检查。
    # 仅检查 memory_server.py 文件存在不足以证明解释器、依赖和协议都可用。
    # inspect 也创建/回收独立进程，不复用一个永久常驻的对话服务进程。
    async def inspect_server(self) -> dict[str, Any]:
        """Perform a real MCP handshake and return the server's advertised tools."""
        process = await self._start()
        try:
            initialize = await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            tools = await self._request(process, "tools/list", {})
            return {"initialize": initialize, "tools": tools.get("tools") or []}
        finally:
            await self._stop(process)

    async def _initialize(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        return await self._request(
            process,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "devflow-web-agent", "version": "0.1.0"},
            },
        )

    # 使用当前 Python 解释器和明确脚本参数启动服务，把 stdin/stdout/stderr 分别接成管道。
    # 数据库 URL 明确传给子进程，保证记忆服务查询同一后端数据库；字符编码设 UTF-8 便于中文协议。
    # AUTO_SYNC_ENABLED 使用 setdefault，仅在环境没有此键时补 false，不应误说成总是强制关闭继承配置。
    # 参数通过 create_subprocess_exec 传递而非 shell 拼接，避免普通参数被解释成 shell 语法。
    async def _start(self) -> asyncio.subprocess.Process:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["DATABASE_URL"] = settings.database_url
        env.setdefault("AUTO_SYNC_ENABLED", "false")
        process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return process

    async def _notify(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> None:
        await self._write(process, {"jsonrpc": "2.0", "method": method, "params": params})

    # 为一次 JSON-RPC 请求分配 ID 并写入 stdin，再在 30 秒内等待 stdout 的下一行。
    # 空行/进程关闭、非法 UTF-8/JSON 和 error 字段分别转换成 McpClientError。
    # 这里按当前顺序协议读取单个响应，不在同一连接上管理多路并发请求和任意服务端通知。
    # 服务端普通日志若写 stdout 会被当成响应解析，所以协议外诊断应走 stderr。
    # 超时后上层 finally 仍会停止进程；不能只抛错误却把独立任务永久留在后台。
    async def _request(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id  # 把当前计数值固定为本次请求身份，随后计数器递增留给下一次请求
        # 先保存本次 ID 再递增，因此此次发送不受计数器新值影响；当前实现按顺序读响应，并未校验返回 ID。
        self._next_id += 1
        await self._write(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if process.stdout is None:
            raise McpClientError("MCP process has no stdout")
        try:
            # 等待一行协议结果并设 30 秒上限，防止子进程无响应把本轮 Agent 永久挂住。
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        except TimeoutError as exc:
            raise McpClientError(f"MCP request timed out: {method}") from exc
        if not line:
            stderr = ""
            if process.stderr is not None:
                stderr = (await process.stderr.read()).decode("utf-8", errors="replace")
            raise McpClientError(f"MCP server closed stdout. {stderr}")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise McpClientError(f"MCP server returned an invalid JSON response for {method}") from exc
        # 协议能解析不代表业务成功，服务端 error 要转换成客户端异常，交给上层记录真实失败。
        if "error" in response:
            error = response["error"]
            raise McpClientError(str(error.get("message") or error))
        return response.get("result") or {}

    # 把一个 JSON-RPC 对象编码为 UTF-8 单行 JSON，并用换行作为当前 stdio 协议帧边界。
    # drain 等待管道缓冲有能力继续写入，不代表服务端已经完成业务处理。
    # 业务完成仍以 _request 读到并解析结果为准，写成功不是调用成功。
    async def _write(self, process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise McpClientError("MCP process has no stdin")
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            process.kill()
        await process.wait()
