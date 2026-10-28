# 仓库接入、同步、选择与删除接口；区分远程托管 checkout 与用户已有本地目录。
# 接入元数据不等于同步完成；删除本地记录与清理磁盘目录也必须区分，不能误删用户工作区。
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, AnalysisResult, AuditLog, ChatFeedback, ChatMessage, ChatSession, CodeRelation, CodeSymbol, Conversation, ConversationMemory, Document, EvidenceItem, Issue, KnowledgeBaseConfig, KnowledgeGraphEdge, KnowledgeSourceDocument, MemoryCandidate, PRFile, PRReviewComment, ProjectIndexState, PullRequest, RecallEvent, Repository, RetrievalTestRun, ThreadMemory, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.repos import RepoConnectRequest, RepoConnectResponse, RepoResponse, RepoSyncRequest, RepoSyncResponse
from app.core.config import settings
from app.core.security import encrypt_token
from app.services.github.client import GitHubClient
from app.services.code_analysis import sync_repository_code_analysis_by_id
from app.services.rag.vector_store import delete_milvus_documents
from app.services.repos_sync import sync_repository

router = APIRouter()

SUPPORTED_CONNECT_PROVIDERS = {"github", "github_compatible"}
GITHUB_HOSTS = {"github.com", "www.github.com"}


@dataclass(frozen=True)
class GitRemoteInfo:
    owner: str
    repo: str
    clone_url: str
    host: str | None = None
    default_branch: str | None = None


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_user_path(raw_path: str, field_name: str) -> Path:
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty.")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _git_output(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="未找到 git 可执行文件，无法检查本地仓库。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=400, detail=f"Failed to inspect local repository: {detail}") from exc
    return result.stdout.strip()


# 把支持的 HTTPS/SSH 等 Git 远端形式拆成 host/owner/repo 等元数据。
# URL 解析只识别身份，不会仅凭字符串确认仓库真实存在、API 兼容或令牌有权限。
# 后续接入逻辑还需按 provider 和用户选择的模式校验，路径不能只靠仓库显示名拼接。
def _parse_git_remote_url(remote_url: str) -> GitRemoteInfo:
    value = remote_url.strip()
    host: str | None = None
    path = ""
    scp_like = re.match(r"^(?:[^@]+@)?(?P<host>[^:]+):(?P<path>.+)$", value)

    if scp_like and "://" not in value:
        host = scp_like.group("host")
        path = scp_like.group("path")
    else:
        parsed = urlparse(value)
        host = parsed.hostname or parsed.netloc or None
        path = parsed.path

    parts = [item.strip() for item in path.replace("\\", "/").strip("/").split("/") if item.strip()]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")

    owner = parts[-2]
    repo = re.sub(r"\.git$", "", parts[-1], flags=re.IGNORECASE)
    if not owner or not repo:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")
    return GitRemoteInfo(owner=owner, repo=repo, clone_url=value, host=host.lower() if host else None)


# 对用户提供目录运行受控 Git 查询，确认它是可使用的工作树并读取 origin 等信息。
# 本地接入应复用用户目录，不把它误当作可任意重克隆/清理的托管缓存。
# 失败时给出明确原因，避免静默新建一个空目录后让页面误以为已有完整源码。
def _inspect_local_repository(raw_path: str) -> tuple[Path, GitRemoteInfo]:
    path = _resolve_user_path(raw_path, "local_path")
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Local repository path does not exist or is not a directory: {path}")

    inside_work_tree = _git_output(["rev-parse", "--is-inside-work-tree"], path).lower()
    if inside_work_tree != "true":
        raise HTTPException(status_code=400, detail=f"Local path is not inside a git repository: {path}")

    repo_root = Path(_git_output(["rev-parse", "--show-toplevel"], path)).resolve()
    remote_url = _git_output(["remote", "get-url", "origin"], repo_root)
    remote_info = _parse_git_remote_url(remote_url)
    current_branch = _git_output(["branch", "--show-current"], repo_root) or None
    return repo_root, GitRemoteInfo(
        owner=remote_info.owner,
        repo=remote_info.repo,
        clone_url=remote_info.clone_url,
        host=remote_info.host,
        default_branch=current_branch,
    )


def _infer_provider(requested_provider: str | None, remote_host: str | None) -> str:
    provider = requested_provider or "github"
    if remote_host and remote_host not in GITHUB_HOSTS and provider == "github":
        return "github_compatible"
    return provider


def _effective_api_base_url(provider: str, raw_base_url: str | None, remote_host: str | None) -> str | None:
    if provider == "github":
        return "https://api.github.com"
    base_url = _clean_text(raw_base_url)
    if base_url:
        return base_url.rstrip("/")
    if remote_host and remote_host not in GITHUB_HOSTS:
        return f"https://{remote_host}/api/v3"
    return None


def _clone_target_from_parent(raw_parent: str, repo_name: str) -> Path:
    parent = _resolve_user_path(raw_parent, "clone_parent_dir")
    if parent.exists() and not parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Download parent path is not a directory: {parent}")

    target = (parent / repo_name).resolve()
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Download target exists and is not a directory: {target}")
    if target.exists() and not (target / ".git").exists() and any(target.iterdir()):
        raise HTTPException(
            status_code=400,
            detail=f"Download target already exists and is not an empty git repository: {target}",
        )
    return target


# 区分远程接入与本地工作树接入，规范 owner/repo/provider/API 地址和 checkout 元数据。
# 远程路径要准备克隆位置，本地路径要检查已有 Git 工作树；两种模式的目录所有权不同。
# 仓库令牌加密后保存，不把明文 key 返回浏览器当作普通展示字段。
# 初次同步是有限覆盖的准备，后续完整同步状态另由 sources/complete 等字段报告。
# 接入成功表示系统已建立仓库关系，不代表所有 Issue/PR/CI、代码和向量都已全部验证就绪。
@router.post("/connect", response_model=RepoConnectResponse)
async def connect_repo(payload: RepoConnectRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> RepoConnectResponse:
    local_path = _clean_text(payload.local_path)
    clone_parent_dir = _clean_text(payload.clone_parent_dir)
    owner = _clean_text(payload.owner)
    repo_name = _clean_text(payload.repo)
    requested_provider = payload.provider or "github"
    token = payload.token or payload.github_token
    checkout_path: Path | None = None
    checkout_mode = "managed"
    remote_info: GitRemoteInfo | None = None

    if local_path:
        checkout_path, remote_info = _inspect_local_repository(local_path)
        owner = remote_info.owner
        repo_name = remote_info.repo
        checkout_mode = "local"
    elif not owner or not repo_name:
        raise HTTPException(status_code=400, detail="Provide either a local repository path or owner/repository.")

    provider = _infer_provider(requested_provider, remote_info.host if remote_info else None)
    base_url = _effective_api_base_url(provider, payload.api_base_url, remote_info.host if remote_info else None)
    if provider not in SUPPORTED_CONNECT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"{provider} real connection is not implemented yet. Use demo mode or github_compatible with a GitHub-compatible REST API.",
        )

    if not local_path and clone_parent_dir:
        checkout_path = _clone_target_from_parent(clone_parent_dir, repo_name or "repository")

    full_name = f"{owner}/{repo_name}"
    effective_demo_mode = payload.demo_mode
    fallback_message: str | None = None
    repo_data = {
        "id": None,
        "description": "Demo repository for DevFlow AI",
        "default_branch": remote_info.default_branch if remote_info and remote_info.default_branch else "main",
        "clone_url": remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git",
    }
    if not payload.demo_mode:
        try:
            repo_data = await GitHubClient(token, base_url).get_repo(owner or "", repo_name or "")
        except Exception as exc:
            if token or not settings.demo_mode:
                raise HTTPException(status_code=400, detail=f"Repository connection failed: {exc}") from exc
            effective_demo_mode = True
            fallback_message = (
                f"GitHub verification failed without a token: {exc}. "
                "Added the repository in local cache mode; add a token or retry later for full GitHub sync."
            )

    repo = db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    if repo is None:
        repo = Repository(owner=owner or "", name=repo_name or "", full_name=full_name)
        db.add(repo)
    repo.owner = owner or repo.owner
    repo.name = repo_name or repo.name
    repo.provider = provider
    repo.api_base_url = base_url
    if token:
        repo.github_token_encrypted = encrypt_token(token, settings.token_encryption_key)
    repo.description = repo_data.get("description")
    repo.default_branch = repo_data.get("default_branch")
    repo.clone_url = repo_data.get("clone_url") or (remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git")
    repo.local_path = str(checkout_path) if checkout_path else None
    repo.checkout_mode = checkout_mode
    repo.github_id = repo_data.get("id")
    db.commit()
    db.refresh(repo)
    sync_result: RepoSyncResponse | None = None
    if not effective_demo_mode:
        sync_result = await sync_repository(
            db,
            repo,
            RepoSyncRequest(
                sync_issues=payload.sync_issues,
                sync_pull_requests=payload.sync_pull_requests,
                sync_workflow_runs=payload.sync_workflow_runs,
                limit=payload.initial_sync_limit,
            ),
        )
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    elif fallback_message:
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return RepoConnectResponse(
        repo_id=repo.id,
        full_name=repo.full_name,
        provider=repo.provider,
        api_base_url=repo.api_base_url,
        clone_url=repo.clone_url,
        local_path=repo.local_path,
        checkout_mode=repo.checkout_mode,
        default_branch=repo.default_branch,
        connected=True,
        demo_mode=effective_demo_mode,
        message=(
            fallback_message
            or (
                "Demo mode only created a local repository record."
                if effective_demo_mode
                else f"Repository connection verified. {sync_result.message if sync_result else 'Sync not started.'}"
            )
        ),
        sync_status=sync_result.status if sync_result else ("fallback" if fallback_message else "skipped"),
        synced=sync_result.synced if sync_result else {"issues": 0, "pull_requests": 0, "workflow_runs": 0},
        sync_details=sync_result.model_dump(mode="json") if sync_result else None,
    )


@router.get("", response_model=list[RepoResponse])
async def list_repos(db: Session = Depends(get_db)) -> list[Repository]:
    return (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )


# 接受本次选择的数据源与主记录总量上限，调用统一 sync_repository，而不另写同步算法。
# 响应中的 synced 只是实际处理数量，sources 才能说明是否全分页、哪里 partial/failed。
# 同步结果与代码分析/索引等后续流程应按实际实现分开判断，不能从一个计数推断全部能力正常。
@router.post("/{repo_id}/sync", response_model=RepoSyncResponse)
async def sync_repo(
    repo_id: UUID,
    payload: RepoSyncRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    db: Session = Depends(get_db),
) -> RepoSyncResponse:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    result = await sync_repository(db, repo, payload)
    if result.status == "failed":
        response.status_code = 502
    background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return result


@router.post("/{repo_id}/select")
async def select_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    repo.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"repo_id": str(repo.id), "full_name": repo.full_name, "selected": True}


# 清理本仓库关联的对话、分析、知识、图谱、评测等本地记录，并处理约定的托管资源。
# 关联记录需要按依赖关系有序清理，避免外键残留或误删另一个仓库的数据。
# 目录清理要区分用户已有本地工作树与系统托管 checkout；数据库 delete 不代表磁盘任意目录可删。
# 向量/文件系统与 SQL 不是同一事务，失败时需要检查具体已清理与未清理范围。
# 这是有破坏性的业务入口，不能在普通查询或仓库切换时顺手执行。
@router.delete("/{repo_id}")
async def delete_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    issue_ids = [item.id for item in db.query(Issue.id).filter(Issue.repo_id == repo.id).all()]
    pr_ids = [item.id for item in db.query(PullRequest.id).filter(PullRequest.repo_id == repo.id).all()]
    run_ids = [item.id for item in db.query(WorkflowRun.id).filter(WorkflowRun.repo_id == repo.id).all()]
    conversation_ids = [item.id for item in db.query(Conversation.id).filter(Conversation.repo_id == repo.id).all()]
    session_ids = [
        item.id
        for item in db.query(ChatSession.id)
        .filter(
            or_(
                ChatSession.repo_id == repo.id,
                ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id)
        .filter(
            or_(
                AgentWorkflowRun.repo_id == repo.id,
                AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]

    for target_type, ids in [
        ("issue", issue_ids),
        ("pull_request", pr_ids),
        ("workflow_run", run_ids),
    ]:
        if ids:
            db.query(AnalysisResult).filter(
                AnalysisResult.target_type == target_type,
                AnalysisResult.target_id.in_(ids),
            ).delete(synchronize_session=False)

    knowledge_sources = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .all()
    )
    stored_paths = [Path(source.stored_path) for source in knowledge_sources]

    if pr_ids:
        db.query(PRFile).filter(PRFile.pr_id.in_(pr_ids)).delete(synchronize_session=False)
        db.query(PRReviewComment).filter(PRReviewComment.pr_id.in_(pr_ids)).delete(synchronize_session=False)
    db.query(Document).filter(Document.repo_id == repo.id).delete(synchronize_session=False)
    db.query(RetrievalTestRun).filter(RetrievalTestRun.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeSourceDocument).filter(KnowledgeSourceDocument.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ActionDraft).filter(ActionDraft.repo_id == repo.id).delete(synchronize_session=False)
    db.query(AuditLog).filter(AuditLog.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(
        or_(
            MemoryCandidate.repo_id == repo.id,
            MemoryCandidate.conversation_id.in_(conversation_ids) if conversation_ids else False,
            MemoryCandidate.session_id.in_(session_ids) if session_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(RecallEvent).filter(
        or_(
            RecallEvent.repo_id == repo.id,
            RecallEvent.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(
        or_(
            AgentWorkflowRun.repo_id == repo.id,
            AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatMessage).filter(
        or_(
            ChatMessage.repo_id == repo.id,
            ChatMessage.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatSession).filter(
        or_(
            ChatSession.repo_id == repo.id,
            ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(
        or_(
            ConversationMemory.repo_id == repo.id,
            ConversationMemory.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(
        or_(
            EvidenceItem.repo_id == repo.id,
            EvidenceItem.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(AgentRun).filter(
        or_(
            AgentRun.repo_id == repo.id,
            AgentRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(Conversation).filter(Conversation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(Issue).filter(Issue.repo_id == repo.id).delete(synchronize_session=False)
    db.query(PullRequest).filter(PullRequest.repo_id == repo.id).delete(synchronize_session=False)
    db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).delete(synchronize_session=False)
    for workspace in db.query(Workspace).all():
        next_repo_ids = [item for item in (workspace.repo_ids or []) if str(item) != str(repo.id)]
        if next_repo_ids != (workspace.repo_ids or []):
            workspace.repo_ids = next_repo_ids
    db.delete(repo)
    db.commit()

    milvus_deleted = delete_milvus_documents(repo_id=repo.id, strict=False)
    removed_files = 0
    file_cleanup_errors: list[str] = []
    for path in stored_paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed_files += 1
        except OSError as exc:
            file_cleanup_errors.append(f"{path}: {exc}")

    return {
        "repo_id": str(repo_id),
        "deleted": True,
        "cleanup": {
            "milvus_deleted": milvus_deleted,
            "removed_files": removed_files,
            "file_errors": file_cleanup_errors,
        },
    }


# 展示最近同步尝试的状态、来源覆盖、错误和时间，不重新向 GitHub 发起全量同步。
# 读取状态与执行同步分离，前端轮询不会因为每次刷新就重复拉取全部远端数据。
# 没有可验证旧同步记录时应承认覆盖未知，不能仅靠 last_sync_at 给“完整”结论。
@router.get("/{repo_id}/sync-status")
async def sync_status(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return {
        "repo_id": str(repo.id),
        "status": (repo.sync_metadata or {}).get("status", "unknown" if repo.last_sync_at else "not_started"),
        "last_sync_at": repo.last_sync_at,
        "message": (repo.sync_metadata or {}).get("message", "缺少同步状态，请重新同步。" if repo.last_sync_at else "仓库还没有同步过。"),
        "details": repo.sync_metadata,
    }
