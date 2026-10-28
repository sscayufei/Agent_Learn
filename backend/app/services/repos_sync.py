# GitHub 同步服务：按远端稳定 ID 更新本地 Issue/PR/CI，再保存逐来源的分页覆盖和失败信息。
# 请求失败与成功但为空必须区分；不能用失败响应清掉已有 PR 文件、评论或 Job 明细。
from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import Issue, PRFile, PRReviewComment, PullRequest, Repository, WorkflowRun
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests


def parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def login_of(user: dict | None) -> str | None:
    return user.get("login") if user else None


# 仓库同步总流程：每个主数据源分别拉取、按远端稳定 ID 更新本地记录，再补 PR/CI 明细。
# sources 的 skipped/empty/completed/partial/failed 描述覆盖；synced 只描述实际处理条数，二者不可互相替代。
# Issue/PR 的 number 是仓库内展示编号，github_*_id 是远端稳定标识，repo.id 是本系统内部仓库 UUID。
# PR 文件/评论只有拉取成功才删除旧明细重建；请求失败保留旧数据，避免网络抖动让数据库突然变空。
# CI 保存 Job/Step，失败运行再取日志，为后续排障保留对应证据。
# 最后根据所选来源汇总状态，只在完整成功/成功为空时更新 last_sync_at，所有尝试保留 attempted_at。
# 关系库提交统一在末尾；远端读取成功不代表后续每个明细也完整，部分失败会降为 partial。
async def sync_repository(db: Session, repo: Repository, payload: RepoSyncRequest) -> RepoSyncResponse:
    if repo.provider not in {"github", "github_compatible"}:
        raise HTTPException(status_code=400, detail=f"{repo.provider} sync is not implemented yet")

    synced = {"issues": 0, "pull_requests": 0, "workflow_runs": 0}  # 本轮各类实际处理条数；同步尚未执行，三类都从 0 计数，不是数据库已有记录总量
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    base_url = repo.api_base_url
    # 先假设三类均未选择，再由实际请求覆盖；未选择不等于请求失败，也不参与“全部失败”的判定。
    sources = {key: {"status": "skipped", "count": 0, "complete": False, "pages": 0} for key in synced}
    errors: list[str] = []  # 本轮主接口/明细请求的失败说明，初始没有已知错误

    # 内部拉取包装，区分主接口与某条主记录的 detail 接口，并统一捕获预期 HTTP/数据异常。
    # 主接口失败标 failed，明细失败让对应来源变 partial；都记录来源/目标和简化原因便于定位。
    # 失败返回 None；成功为空返回空列表，这个语义区别决定后面能否安全替换旧明细。
    # 成功分页结果还携带 complete/pages，达到限量但没抓完必须标 partial，而不是“处理了 50 条就完成”。
    # 此包装不负责重试所有错误，也不把异常转换成一份假成功的空数据。
    async def fetch(source: str, loader, *args, detail: str | None = None, **kwargs):
        try:
            items = await loader(*args, **kwargs)
        except (httpx.HTTPError, ValueError) as exc:
            reason = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            message = f"{source}{'/' + detail if detail else ''}: {reason}"
            errors.append(message)
            sources[source]["status"] = "partial" if detail else "failed"
            sources[source]["complete"] = False
            return None
        # 只有主列表结果直接定义来源覆盖状态，某条明细失败只降低其所属来源完整性，不重置主列表数量。
        if detail is None:
            # 分页结果可携带覆盖信息；普通测试列表没有该属性时沿用兼容默认，真实分页用 PaginatedItems。
            complete = getattr(items, "complete", True)
            sources[source] = {
                "status": ("completed" if items else "empty") if complete else "partial",
                "count": len(items), "complete": complete, "pages": getattr(items, "pages", 1),
            }
        return items

    if payload.sync_issues:
        remote_issues = await fetch("issues", list_issues, repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        # 每条远端 Issue 独立查找/新建/更新，遍历结束后累计本轮数量；请求失败的 None 不会产生新行。
        for item in remote_issues or []:
            # 同一远端对象多次同步应更新同一行；不能只按标题去重，标题会修改且不同 Issue 可能重名。
            issue = db.query(Issue).filter(Issue.repo_id == repo.id, Issue.github_issue_id == item.get("id")).one_or_none()
            if issue is None:
                issue = Issue(repo_id=repo.id, github_issue_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(issue)
            issue.number = item.get("number", issue.number)
            issue.title = item.get("title") or issue.title
            issue.body = item.get("body")
            issue.state = item.get("state") or "open"
            issue.labels = [label.get("name") for label in item.get("labels", []) if label.get("name")]
            issue.author = login_of(item.get("user"))
            issue.assignees = [login_of(user) for user in item.get("assignees", []) if login_of(user)]
            issue.created_at = parse_github_datetime(item.get("created_at"))
            issue.updated_at = parse_github_datetime(item.get("updated_at"))
            issue.closed_at = parse_github_datetime(item.get("closed_at"))
            synced["issues"] += 1

    if payload.sync_pull_requests:
        remote_prs = await fetch("pull_requests", list_pull_requests, repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        # PR 除主记录外还需要同步文件和评论，所以逐条先获得真实本地 pr.id，再关联其明细。
        for item in remote_prs or []:
            pr = db.query(PullRequest).filter(PullRequest.repo_id == repo.id, PullRequest.github_pr_id == item.get("id")).one_or_none()
            if pr is None:
                pr = PullRequest(repo_id=repo.id, github_pr_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(pr)
                # PR 新行需先取得内部主键，后续 PRFile/PRReviewComment 用它建关联；flush 尚不提交整轮同步。
                db.flush()
            pr.number = item.get("number", pr.number)
            pr.title = item.get("title") or pr.title
            pr.body = item.get("body")
            pr.state = item.get("state") or "open"
            pr.author = login_of(item.get("user"))
            pr.base_branch = (item.get("base") or {}).get("ref")
            pr.head_branch = (item.get("head") or {}).get("ref")
            pr.merged_at = parse_github_datetime(item.get("merged_at"))
            pr.created_at = parse_github_datetime(item.get("created_at"))
            pr.updated_at = parse_github_datetime(item.get("updated_at"))

            files = await fetch("pull_requests", list_pull_request_files, repo.owner, repo.name, pr.number, token=token, base_url=base_url, detail=f"#{pr.number}/files")
            # None 表示拉取失败，必须保留旧明细；空列表表示远端成功返回空，允许清空旧明细。
            if files is not None:
                db.query(PRFile).filter(PRFile.pr_id == pr.id).delete()
            # 一条远端文件对应一条 PRFile，后续审查才能按文件定位；不把所有补丁合成一个失去路径的字段。
            for file_item in files or []:
                db.add(
                    PRFile(
                        pr_id=pr.id,
                        filename=file_item.get("filename") or "",
                        status=file_item.get("status") or "modified",
                        additions=file_item.get("additions") or 0,
                        deletions=file_item.get("deletions") or 0,
                        patch=file_item.get("patch"),
                    )
                )
            review_comments = await fetch("pull_requests", list_pull_request_review_comments, repo.owner, repo.name, pr.number, token=token, base_url=base_url, detail=f"#{pr.number}/comments")
            # 与文件明细一样，None 是失败保护，空列表是远端确实无评论；不能写成 if review_comments 后遗漏清空语义。
            if review_comments is not None:
                db.query(PRReviewComment).filter(PRReviewComment.pr_id == pr.id).delete()
            # 逐条保存 Review 评论的远端身份与位置，便于回查哪份文件、哪一行被讨论。
            for comment_item in review_comments or []:
                db.add(
                    PRReviewComment(
                        pr_id=pr.id,
                        github_comment_id=comment_item.get("id"),
                        body=comment_item.get("body"),
                        path=comment_item.get("path"),
                        line=comment_item.get("line") or comment_item.get("original_line"),
                        author=login_of(comment_item.get("user")),
                        created_at=parse_github_datetime(comment_item.get("created_at")),
                        updated_at=parse_github_datetime(comment_item.get("updated_at")),
                    )
                )
            synced["pull_requests"] += 1

    if payload.sync_workflow_runs:
        remote_runs = await fetch("workflow_runs", list_workflow_runs, repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        # 逐个更新流水线状态及可用日志，同一远端 run 的重复同步更新已有行，而不是不断插入副本。
        for item in remote_runs or []:
            run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.github_run_id == item.get("id")).one_or_none()
            if run is None:
                run = WorkflowRun(repo_id=repo.id, github_run_id=item.get("id"), name=item.get("name") or "workflow")
                db.add(run)
            run.name = item.get("name") or run.name
            run.status = item.get("status") or "unknown"
            run.conclusion = item.get("conclusion")
            run.html_url = item.get("html_url")
            run.created_at = parse_github_datetime(item.get("created_at"))
            run.updated_at = parse_github_datetime(item.get("updated_at"))
            if run.github_run_id:
                jobs = await fetch("workflow_runs", list_workflow_run_jobs, repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url, detail=f"{run.github_run_id}/jobs")
                if jobs is not None:
                    run.jobs = [
                        {
                            "id": job.get("id"),
                            "name": job.get("name"),
                            "status": job.get("status"),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url"),
                            "started_at": job.get("started_at"),
                            "completed_at": job.get("completed_at"),
                            "steps": [
                                {
                                    "name": step.get("name"),
                                    "status": step.get("status"),
                                    "conclusion": step.get("conclusion"),
                                    "number": step.get("number"),
                                }
                                for step in job.get("steps", [])
                            ],
                        }
                        for job in jobs
                    ]
            # 只为失败 CI 下载日志，减少成功运行的大量无关文本和网络成本，状态本身仍保存在结构化记录里。
            if run.conclusion == "failure" and run.github_run_id:
                logs = await fetch("workflow_runs", get_workflow_run_logs, repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url, detail=f"{run.github_run_id}/logs")
                if logs is not None:
                    run.logs_text = logs
            synced["workflow_runs"] += 1

    # 汇总只看这次选择的来源；用户没勾选 CI 时，不应因 CI skipped 把完整 Issue 同步降级为失败。
    requested = [item for item in sources.values() if item["status"] != "skipped"]
    if not requested:
        status = "skipped"
    # 所选主接口全失败才是 failed；仍有部分有效数据时用 partial，给学员真实覆盖边界。
    elif all(item["status"] == "failed" for item in requested):
        status = "failed"
    elif any(item["status"] in {"failed", "partial"} for item in requested):
        status = "partial"
    else:
        status = "completed" if any(synced.values()) else "empty"
    now = datetime.now(timezone.utc)
    messages = {
        "completed": "已完成所选数据源的全部分页同步。",
        "empty": "所选数据源请求成功，但未返回记录；不能据此判断仓库健康。",
        "partial": "同步不完整：存在请求失败或数量上限，请查看各数据源状态。",
        "failed": "同步失败，未获取到有效的新数据；保留已有数据。",
        "skipped": "未选择数据源，没有执行同步。",
    }
    # 只有完整获取所选来源才刷新成功时间，partial/failed 不能掩盖数据陈旧。
    if status in {"completed", "empty"}:
        repo.last_sync_at = now
    result = RepoSyncResponse(repo_id=repo.id, status=status, synced=synced, sources=sources, errors=errors, message=messages[status])
    # 成功时间与尝试时间分开：失败尝试不能把旧数据包装成刚刚完成的新鲜证据。
    repo.sync_metadata = {**result.model_dump(mode="json"), "attempted_at": now.isoformat(), "limit": payload.limit}
    db.flush()
    db.commit()
    return result


# 为后台调度重新创建 Session，按 ID 获取本仓库，并复用同一个同步实现。
# 不能把前台请求中已关闭的 Session/ORM 对象直接传给长期后台循环继续使用。
# 仓库已删除则返回 None，finally 释放本次连接，避免一次轮询泄漏一个 Session。
async def sync_repository_by_id(repo_id: UUID, payload: RepoSyncRequest) -> RepoSyncResponse | None:
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        return await sync_repository(db, repo, payload)
    finally:
        db.close()
