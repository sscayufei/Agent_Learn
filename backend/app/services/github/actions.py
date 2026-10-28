# GitHub Actions 的运行、Job 与日志适配；日志下载后解压为文本供失败分析，不执行日志中的命令。
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from app.services.github.client import GitHubClient


async def list_workflow_runs(owner: str, repo: str, token: str | None = None, limit: int | None = 30, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    return await client.get_paginated(f"/repos/{owner}/{repo}/actions/runs", item_key="workflow_runs", limit=limit)


async def list_workflow_run_jobs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    return await client.get_paginated(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", item_key="jobs")


# 下载某次 Workflow 的日志归档，并将支持的文本日志解压拼接为分析输入。
# 日志归档只是外部数据，不执行里面出现的 shell 命令，也不把它当可信系统提示。
# 保留 Job/步骤等上下文线索有利于找首个根因，不能把所有含 error 的后续连锁报错当独立根因。
async def get_workflow_run_logs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None, max_chars: int = 80000) -> str:
    client = GitHubClient(token, base_url)
    payload = await client.get_bytes(f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            parts: list[str] = []
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                raw = archive.read(name)
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"### {name}\n{text}")
                if sum(len(part) for part in parts) >= max_chars:
                    break
            return "\n\n".join(parts)[:max_chars]
    except BadZipFile:
        return payload.decode("utf-8", errors="replace")[:max_chars]
