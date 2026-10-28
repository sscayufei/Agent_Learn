# 外部写操作的人工确认边界：先创建 pending_confirmation 草稿，再由有权限的人确认执行。
# 执行结果与失败都会留审计；分析 Agent 生成草稿不等于已经在 GitHub 发布评论/关闭 Issue。
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import ActionDraft, AuditLog, Issue, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.action_drafts import ActionDraftConfirmResponse, ActionDraftCreateRequest, ActionDraftResponse, AuditLogResponse
from app.services.github.client import GitHubClient
from app.services.permissions import require_permission, write_audit_log

router = APIRouter()


# 优先从草稿元数据读取外部目标编号，缺失时再按内部 target_id 查 Issue/PR 记录。
# 内部 UUID 无法直接传给 GitHub 的 /issues/{number} 接口，需要这一层标识转换。
# 非数字或查不到返回 None，由具体动作给出缺少目标的错误，而不是默认为 0 或最新 Issue。
def _target_number(db: Session, draft: ActionDraft) -> int | None:
    meta = draft.meta or {}
    raw = meta.get("target_number") or meta.get("issue_number") or meta.get("pr_number")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if draft.target_id and draft.target_type == "issue":
        issue = db.get(Issue, draft.target_id)
        return issue.number if issue else None
    if draft.target_id and draft.target_type in {"pull_request", "pr"}:
        pr = db.get(PullRequest, draft.target_id)
        return pr.number if pr else None
    return None


# 把已进入确认流程的 draft_type 映射到有限的 GitHub 写方法。
# 评论、建 Issue、关 Issue、加标签分别验证所需编号/标题/标签等字段，未知类型明确报不支持。
# send_report 当前只是本系统记录为发送，不代表接通了外部消息渠道，代码的返回说明要如实理解。
# 它不负责批准草稿；调用方应先校验 draft 状态、审批权限与所属仓库。
# 外部 HTTP 请求一旦成功，后续 SQL 出错也无法靠 rollback 撤销远端动作。
async def _execute_github_draft(db: Session, draft: ActionDraft, repo: Repository) -> dict:
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    client = GitHubClient(token=token, base_url=repo.api_base_url)
    draft_type = draft.draft_type.lower()
    issue_number = _target_number(db, draft)
    # GitHub 的 Issue 评论接口也能覆盖 PR 会话评论，具体行内 Review 评论属于不同能力，不能混为一谈。
    if draft_type in {"issue_comment", "pr_comment", "comment"}:
        if issue_number is None:
            raise RuntimeError("评论草稿缺少 issue/pr number")
        return await client.create_issue_comment(repo.owner, repo.name, issue_number, draft.content)
    if draft_type == "create_issue":
        title = draft.title or (draft.content.splitlines()[0][:120] if draft.content else "DevFlow AI Issue")
        return await client.create_issue(repo.owner, repo.name, title, draft.content)
    if draft_type == "close_issue":
        if issue_number is None:
            raise RuntimeError("关闭 Issue 草稿缺少 issue number")
        return await client.close_issue(repo.owner, repo.name, issue_number)
    if draft_type in {"label", "add_label", "add_labels"}:
        if issue_number is None:
            raise RuntimeError("Label 草稿缺少 issue/pr number")
        labels = draft.meta.get("labels") if draft.meta else None
        if not isinstance(labels, list) or not labels:
            labels = [draft.title] if draft.title else []
        if not labels:
            raise RuntimeError("Label 草稿缺少 labels")
        return await client.add_issue_labels(repo.owner, repo.name, issue_number, [str(item) for item in labels])
    if draft_type == "send_report":
        return {"status": "recorded", "message": "报告草稿已在 DevFlow AI 中标记为已发送，未接入外部消息渠道。"}
    raise RuntimeError(f"暂不支持执行草稿类型：{draft.draft_type}")


# 保存待确认草稿，记录创建人、目标、内容、风险和元数据，并写创建审计。
# 这里没有调用 _execute_github_draft，所以分析 Agent 生成草稿不会直接发 GitHub 评论。
# db.commit 同时提交本地草稿与审计，refresh 读取数据库默认值后返回展示对象。
# 风险等级是业务提示字段，不能仅凭客户端自报 low 就取消后续审批权限要求。
@router.post("", response_model=ActionDraftResponse)
async def create_action_draft(
    payload: ActionDraftCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    draft = ActionDraft(
        repo_id=payload.repo_id,
        draft_type=payload.draft_type,
        target_type=payload.target_type,
        target_id=payload.target_id,
        title=payload.title,
        content=payload.content,
        risk_level=payload.risk_level,
        created_by=user.id,
        meta=payload.metadata,
    )
    db.add(draft)
    write_audit_log(
        db,
        user=user,
        repo_id=payload.repo_id,
        action="action_draft.create",
        target_type=payload.target_type,
        target_id=str(payload.target_id) if payload.target_id else None,
        request_json=payload.model_dump(mode="json"),
    )
    db.commit()
    db.refresh(draft)
    return draft


@router.get("", response_model=list[ActionDraftResponse])
async def list_action_drafts(
    repo_id: UUID | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[ActionDraft]:
    query = db.query(ActionDraft)
    if repo_id:
        query = query.filter(ActionDraft.repo_id == repo_id)
    if status:
        query = query.filter(ActionDraft.status == status)
    return query.order_by(ActionDraft.created_at.desc()).limit(100).all()


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    repo_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("settings:manage")),
) -> list[AuditLog]:
    query = db.query(AuditLog)
    if repo_id:
        query = query.filter(AuditLog.repo_id == repo_id)
    return query.order_by(AuditLog.created_at.desc()).limit(100).all()


@router.get("/{draft_id}", response_model=ActionDraftResponse)
async def get_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> ActionDraft:
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    return draft


# 只允许取消 pending_confirmation 状态，防止把已经执行的动作伪装成“取消了”。
# 取消写本地状态与审计，不会反向撤销已发生的 GitHub 操作。
# 若需撤销已发布评论/标签等，应设计明确的对应动作，而不是复用这里改状态。
@router.post("/{draft_id}/cancel", response_model=ActionDraftResponse)
async def cancel_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以取消")
    draft.status = "cancelled"
    write_audit_log(db, user=user, repo_id=draft.repo_id, action="action_draft.cancel", target_type=draft.target_type, target_id=str(draft.id))
    db.commit()
    db.refresh(draft)
    return draft


# 取得审批身份，确认草稿仍待确认且仓库存在，才发起具体外部动作。
# 成功写 executed、确认人、执行时间和远端结果；异常写 failed 与错误，再分别记录审计。
# 失败也返回可回查草稿，避免只抛 HTTP 错误却没有业务状态记录。
# pending 检查是流程前置条件，不是数据库行锁/分布式幂等键；不要据此声称并发重复确认绝对不可能。
# 远端动作与本地事务存在边界，人工重试前要结合 execution_result/audit 核对是否已经生效。
@router.post("/{draft_id}/confirm", response_model=ActionDraftConfirmResponse)
async def confirm_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:approve")),
) -> dict:
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以执行")
    repo = db.get(Repository, draft.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    try:
        result = await _execute_github_draft(db, draft, repo)
        # 只有外部调用正常返回才更新成功状态，先写成功再调用会在异常时误导界面。
        draft.status = "executed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.execution_result = result
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="success",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json=result if isinstance(result, dict) else {"result": result},
        )
    except Exception as exc:
        # 保存失败现场便于复核；网络超时可能仍需核对远端是否已处理，不能直接断言没有任何副作用。
        draft.status = "failed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.error_message = str(exc)
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="failed",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json={"error": str(exc)},
        )
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return {"draft": draft, "audit_id": audit.id}
