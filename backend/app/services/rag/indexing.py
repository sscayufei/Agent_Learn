# 把 GitHub 的非结构化内容转成 RAG 文档：Issue 描述、PR 说明/评论、失败 CI 日志等。
# 状态、成员等结构化事实仍查数据库；CI 日志在切片前先清理噪声与敏感模式。
import re

from sqlalchemy.orm import Session, selectinload

from app.db.models import Issue, KnowledgeBaseConfig, PullRequest, Repository, WorkflowRun
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.vector_store import DocumentPayload, replace_documents

SYNCED_REPO_SOURCE_TYPES = ["issue", "pull_request", "workflow_run"]
CI_LOG_CHUNK_CHARS = 8000
CI_LOG_MAX_CHUNKS = 6


# 删除终端控制字符和常见噪声，按已知模式替换令牌/口令等敏感片段，再用于知识检索。
# CI 日志属于外部输入，不能把其中看似命令/指令的文字当作当前 Agent 的执行要求。
# 规则清洗只能覆盖已知模式，运行方仍需从源头避免把真实秘密打印进日志。
def sanitize_ci_log(text: str) -> str:
    sanitized = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    sanitized = re.sub(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED_GITHUB_TOKEN]", sanitized)
    sanitized = re.sub(r"\bAKIA[A-Z0-9]{16}\b", "[REDACTED_AWS_ACCESS_KEY]", sanitized)
    sanitized = re.sub(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        "[REDACTED_JWT]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(authorization\s*:\s*bearer)\s+[^\s,;]+",
        r"\1 [REDACTED]",
        sanitized,
    )
    sanitized = re.sub(r"(?i)(::add-mask::)[^\r\n]+", r"\1[REDACTED]", sanitized)
    return re.sub(
        r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|"
        r"token|secret|password|passwd|private[_-]?key|client[_-]?secret|account[_-]?key)[a-z0-9_]*)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        sanitized,
    )


def _ci_log_chunks(text: str) -> tuple[list[str], bool]:
    sanitized = sanitize_ci_log(text.strip())
    max_chars = CI_LOG_CHUNK_CHARS * CI_LOG_MAX_CHUNKS
    bounded = sanitized[:max_chars]
    chunks = [
        bounded[start : start + CI_LOG_CHUNK_CHARS]
        for start in range(0, len(bounded), CI_LOG_CHUNK_CHARS)
        if bounded[start : start + CI_LOG_CHUNK_CHARS].strip()
    ]
    return chunks, len(sanitized) > max_chars


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _payload(source_type: str, source_id, title: str, content: str, metadata: dict) -> DocumentPayload | None:
    if not content.strip():
        return None
    return DocumentPayload(
        source_type=source_type,
        source_id=source_id,
        title=title[:500],
        content=content,
        metadata=metadata,
    )


# 从当前仓库的结构化记录生成适合 RAG 的非结构化文档投影。
# Issue 以描述为主，PR 以说明/评论为主，CI 以失败日志为主；状态与统计仍留在结构化查询路径。
# 每条 payload 保存来源 UUID、显示标题和更新时间等，检索答案可以沿这些信息回查原记录。
# 按有效模型契约生成向量，再与文档写入/来源替换流程协作；开关参数允许重建时分阶段处理。
# 不能因为本地保存了 PRFile 或团队资料，就把所有数据库字段都无条件投入向量库。
def index_repository_documents(
    db: Session,
    repo_id,
    *,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> int:
    repo = db.get(Repository, repo_id)
    if repo is None:
        return 0

    payloads: list[DocumentPayload] = []

    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    for issue in issues:
        content = "\n".join([issue.title, issue.body or ""])
        item = _payload(
            "issue",
            issue.id,
            issue.title,
            content,
            {
                "number": issue.number,
                "labels": issue.labels or [],
                "state": issue.state,
                "author": issue.author,
                "assignees": issue.assignees or [],
                "created_at": _iso(issue.created_at),
                "updated_at": _iso(issue.updated_at),
                "closed_at": _iso(issue.closed_at),
            },
        )
        if item:
            payloads.append(item)

    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    for pr in prs:
        comments_text = "\n".join(
            f"{comment.path or ''}:{comment.line or ''} {comment.author or ''}: {comment.body or ''}"
            for comment in pr.review_comments
        )
        content = "\n".join(
            [
                pr.title,
                pr.body or "",
                "review comments:",
                comments_text,
            ]
        )
        item = _payload(
            "pull_request",
            pr.id,
            pr.title,
            content,
            {
                "number": pr.number,
                "state": pr.state,
                "author": pr.author,
                "files": [file.filename for file in pr.files],
                "base_branch": pr.base_branch,
                "head_branch": pr.head_branch,
                "created_at": _iso(pr.created_at),
                "updated_at": _iso(pr.updated_at),
                "merged_at": _iso(pr.merged_at),
            },
        )
        if item:
            payloads.append(item)

    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    for run in runs:
        if run.conclusion != "failure" or not (run.logs_text or "").strip():
            continue
        log_chunks, logs_truncated = _ci_log_chunks(run.logs_text or "")
        for chunk_index, log_chunk in enumerate(log_chunks):
            content = "\n".join([run.name, log_chunk])
            item = _payload(
                "workflow_run",
                run.id,
                run.name if len(log_chunks) == 1 else f"{run.name}#{chunk_index + 1}",
                content,
                {
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "html_url": run.html_url,
                    "jobs": [job.get("name") for job in (run.jobs or []) if isinstance(job, dict)],
                    "created_at": _iso(run.created_at),
                    "updated_at": _iso(run.updated_at),
                    "chunk_index": chunk_index,
                    "total_chunks": len(log_chunks),
                    "logs_truncated": logs_truncated,
                    "secrets_redacted": True,
                },
            )
            if item:
                payloads.append(item)

    vectors = None
    embedding_info = None
    config = (
        db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_id).one_or_none()
        if generate_embeddings
        else None
    )
    if config is not None and payloads:
        vectors = embed_knowledge_texts(
            [f"{payload.title}\n{payload.content}" for payload in payloads],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        embedding_info = {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions,
        }

    docs = replace_documents(
        db,
        repo_id,
        SYNCED_REPO_SOURCE_TYPES,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    return len(docs)
