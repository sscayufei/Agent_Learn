#!/usr/bin/env python
# stdio MCP 记忆服务：声明工具 schema，解析 JSON-RPC 请求，再调用仓库/会话范围内的 MemoryTools。
# 它运行在独立进程，需自行创建数据库会话；协议结果与普通日志必须分开，不能污染 stdout。
import asyncio
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import Conversation, Repository  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.services.chat_memory import MemoryTools  # noqa: E402
from app.services.conversations import ensure_conversation  # noqa: E402


SERVER_NAME = "devflow-memory-mcp"
SERVER_VERSION = "0.1.0"

for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _json_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "devflow_search_evidence",
        "description": (
            "搜索 DevFlow 仓库知识和持久记忆。"
            "当需要回忆历史 Issue/PR、失败 CI、上传知识、项目文档、会话摘要、Agent 运行记录或已批准决策时，优先使用这个工具。"
            "当前代码、团队成员和实时状态应使用工作区或结构化工具。"
        ),
        "inputSchema": _json_schema(
            {
                "query": {"type": "string", "description": "搜索查询"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最大结果数"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["query", "repo_id"],
        ),
    },
    {
        "name": "devflow_get_thread_context",
        "description": (
            "读取 DevFlow 仓库会话的完整原始聊天转录，可跨越 Compact Boundary。"
            "支持关键词、消息 UUID、前后范围和字符偏移分页；当用户询问之前说过什么或需要逐字核实时使用。"
        ),
        "inputSchema": _json_schema(
            {
                "keyword": {"type": "string", "description": "可选关键词过滤"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "最大消息数"},
                "message_id": {"type": "string", "description": "可选消息 UUID；用于分页读取一条精确原文。"},
                "before_message_id": {"type": "string", "description": "可选消息 UUID；只读取它之前的记录。"},
                "after_message_id": {"type": "string", "description": "可选消息 UUID；只读取它之后的记录。"},
                "offset": {"type": "integer", "minimum": 0, "description": "读取单条消息时的字符偏移量。"},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 16000, "description": "本次最多返回的正文字符数。"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["repo_id"],
        ),
    },
    {
        "name": "devflow_read_session_events",
        "description": (
            "读取已封存的 DevFlow 聊天会话、摘要、原始消息和工具事件。"
            "在 search_evidence 之后，或需要会话级回忆时使用。"
        ),
        "inputSchema": _json_schema(
            {
                "session_id": {"type": "string", "description": "可选会话片段 UUID，默认使用最近会话。"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最大会话数"},
                "include_transcript": {"type": "boolean", "description": "是否返回封存的原始消息和工具事件，默认 true。"},
                "max_chars": {"type": "integer", "minimum": 400, "maximum": 20000, "description": "本次最多返回的正文字符数。"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["repo_id"],
        ),
    },
]


class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _content(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _resolve_repo(db: Session, args: dict[str, Any]) -> Repository:
    repo_id = args.get("repo_id")
    if not repo_id:
        raise McpError(-32602, "repo_id 为必填项")
    try:
        repo = db.get(Repository, uuid.UUID(str(repo_id)))
    except ValueError as exc:
        raise McpError(-32602, f"repo_id 无效：{repo_id}") from exc
    if repo is None:
        raise McpError(-32602, f"未找到仓库：{repo_id}")
    return repo


def _resolve_conversation(db: Session, repo: Repository, args: dict[str, Any]) -> Conversation:
    conversation_id = args.get("conversation_id")
    if conversation_id:
        try:
            return ensure_conversation(db, repo.id, uuid.UUID(str(conversation_id)))
        except ValueError as exc:
            raise McpError(-32602, f"conversation_id 无效：{conversation_id}") from exc
    return ensure_conversation(db, repo.id)


def _repo_label(repo: Repository) -> str:
    return repo.full_name


async def _call_memory_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    Base.metadata.create_all(bind=engine)
    # 服务是独立进程，没有 Web 请求传进来的会话，所以每次工具调用自行取得数据库 Session。
    db = SessionLocal()
    try:
        # 先按明确 repo_id 找仓库，再限定会话，不能默认读取整个系统所有记忆。
        repo = _resolve_repo(db, args)
        conversation = _resolve_conversation(db, repo, args)
        tools = MemoryTools(db, repo, conversation)

        # 按已知工具名分派到受控方法，传来的名字不能当 Python 函数名动态执行。
        if name == "devflow_search_evidence":
            query = str(args.get("query") or "").strip()
            if not query:
                raise McpError(-32602, "query 为必填项")
            result = await tools.search_evidence({"query": query, "limit": int(args.get("limit") or 5)}, {})
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        if name == "devflow_get_thread_context":
            result = await tools.get_thread_context(
                {
                    "keyword": str(args.get("keyword") or ""),
                    "limit": int(args.get("limit") or 30),
                    "message_id": args.get("message_id"),
                    "before_message_id": args.get("before_message_id"),
                    "after_message_id": args.get("after_message_id"),
                    "offset": int(args.get("offset") or 0),
                    "max_chars": int(args.get("max_chars") or 6000),
                },
                {},
            )
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        if name == "devflow_read_session_events":
            result = await tools.read_session_events(
                {
                    "session_id": args.get("session_id"),
                    "limit": int(args.get("limit") or 5),
                    "include_transcript": bool(args.get("include_transcript", True)),
                    "max_chars": int(args.get("max_chars") or 8000),
                },
                {},
            )
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        raise McpError(-32601, f"未知工具：{name}")
    finally:
        # 无论参数错误、查询失败还是正常返回，都释放本次会话，避免多次工具调用泄漏连接。
        db.close()


def _initialize() -> dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {"listChanged": False}},
    }


async def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")  # 原请求身份，响应原样带回；通知可能没有 id，通知路径不应强行制造回复
    method = request.get("method")  # 决定执行初始化、列工具还是调工具，不是业务查询正文
    params = request.get("params") or {}  # 协议参数对象，缺省时用空对象；具体 arguments 还要在调用分支校验

    # 这是客户端完成初始化的通知，按当前协议无需响应，返回 None 防止客户端把多余行当下个请求结果。
    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = _initialize()
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            # 模型/客户端传来的工具参数必须是对象，类型不对要返回协议错误，不能直接当关键字参数继续用。
            if not isinstance(arguments, dict):
                raise McpError(-32602, "arguments 必须是对象")
            result = await _call_memory_tool(name, arguments)
        else:
            raise McpError(-32601, f"未找到方法：{method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except McpError as exc:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": exc.message}}
    except Exception as exc:
        # 调试堆栈写 stderr，stdout 必须只保留单行 JSON 协议内容，否则客户端无法解帧。
        print(traceback.format_exc(), file=sys.stderr)
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": str(exc)}}


async def main() -> None:
    print(f"[{SERVER_NAME}] 正在 stdio 上运行", file=sys.stderr)
    # 一个服务进程可顺序处理多条请求，持续读 stdin 直到父进程关闭管道；不是读取一次 initialize 就退出。
    while True:
        # stdin.readline 会阻塞，把它放在线程里等待，避免占住异步事件循环。
        line = await asyncio.to_thread(sys.stdin.readline)
        # 读到 EOF 表示输入管道关闭，终止服务；这是结束，不是需要等待的普通空白请求。
        if not line:
            break
        line = line.strip()
        # 去空白后为空则跳过这一行，与上面的 EOF 不同，服务仍继续接收后续请求。
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        else:
            response = await handle_request(request)
        # 通知返回 None 时不输出；普通请求才打印单行 JSON，并立即 flush 让父进程读到结果。
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
