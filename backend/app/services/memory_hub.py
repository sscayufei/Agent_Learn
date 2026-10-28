# 长期记忆治理：自动抽取先形成候选，人工批准后才成为可用记忆；召回记录提供可观察性。
# 候选、已批准知识和原始聊天记录不是同一种数据，拒绝候选不会抹除原聊天。
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import (
    Document,
    EvidenceItem,
    MemoryCandidate,
    ProjectIndexState,
    RecallEvent,
    Repository,
)
from app.services.rag.vector_store import add_document, milvus_runtime_status


# 保存真实检索/回忆工具的查询、范围、候选与引用，形成可回看的 RecallEvent。
# 召回记录是观察日志，不把候选自动升级为正式长期知识。
# 前端可以据此比较预检索、改写查询与精确原文读取，回答中的“我记得”需有来源可查。
def record_recall_event(
    db: Session,
    *,
    repo_id: uuid.UUID | None,
    conversation_id: uuid.UUID | None,
    tool_name: str,
    query: str,
    results: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    scope: str = "project",
    mode: str = "hybrid",
    status: str = "success",
    metadata: dict[str, Any] | None = None,
) -> RecallEvent:
    safe_results = [_recall_result_payload(item) for item in (results or [])[:12]]
    event = RecallEvent(
        repo_id=repo_id,
        conversation_id=conversation_id,
        tool_name=tool_name,
        query=query.strip(),
        scope=scope,
        mode=mode,
        status=status,
        result_count=len(results or []),
        results_json=safe_results,
        citations_json=(citations or [])[:12],
        meta=metadata or {},
    )
    db.add(event)
    db.flush()
    return event


def list_recall_events(
    db: Session,
    *,
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    query = db.query(RecallEvent).filter(RecallEvent.repo_id == repo_id)
    if conversation_id:
        query = query.filter(RecallEvent.conversation_id == conversation_id)
    rows = query.order_by(RecallEvent.created_at.desc()).limit(min(max(limit, 1), 100)).all()
    return [_recall_event_payload(row) for row in rows]


def memory_status(db: Session, repo: Repository) -> dict[str, Any]:
    index_state = db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).one_or_none()
    source_counts = dict(
        db.query(Document.source_type, func.count(Document.id))
        .filter(Document.repo_id == repo.id)
        .group_by(Document.source_type)
        .all()
    )
    evidence_counts = dict(
        db.query(EvidenceItem.source_type, func.count(EvidenceItem.id))
        .filter(EvidenceItem.repo_id == repo.id)
        .group_by(EvidenceItem.source_type)
        .all()
    )
    candidate_counts = dict(
        db.query(MemoryCandidate.status, func.count(MemoryCandidate.id))
        .filter(MemoryCandidate.repo_id == repo.id)
        .group_by(MemoryCandidate.status)
        .all()
    )
    latest_recall = (
        db.query(RecallEvent)
        .filter(RecallEvent.repo_id == repo.id)
        .order_by(RecallEvent.created_at.desc())
        .first()
    )
    return {
        "repo_id": repo.id,
        "index": {
            "status": index_state.status if index_state else "missing",
            "docs_indexed": index_state.docs_indexed if index_state else 0,
            "docs_total": index_state.docs_total if index_state else 0,
            "last_scan_at": index_state.last_scan_at.isoformat() if index_state and index_state.last_scan_at else None,
            "error_message": index_state.error_message if index_state else None,
            "summary_json": index_state.summary_json if index_state else None,
        },
        "documents": {
            "total": sum(int(value) for value in source_counts.values()),
            "by_source_type": source_counts,
        },
        "evidence": {
            "total": sum(int(value) for value in evidence_counts.values()),
            "by_source_type": evidence_counts,
        },
        "recall": {
            "total": db.query(RecallEvent).filter(RecallEvent.repo_id == repo.id).count(),
            "latest_at": latest_recall.created_at.isoformat() if latest_recall and latest_recall.created_at else None,
        },
        "candidates": {
            "total": sum(int(value) for value in candidate_counts.values()),
            "by_status": candidate_counts,
            "pending": int(candidate_counts.get("pending") or 0),
        },
        "vector": milvus_runtime_status(),
    }


# 从一次记忆提取中的 decisions/facts/tasks/preferences/repo_context 生成候选。
# 空内容与已有同类同内容候选跳过，减少反复对话产生大量重复审批项。
# 新项一律 pending，并关联 repo/conversation/session 与捕获时间，保留来处。
# 这里仅 flush，不替上层提交整轮事务，也不自动将未确认提取写进正式 RAG 知识。
def capture_memory_candidates(
    db: Session,
    *,
    repo_id: uuid.UUID | None,
    conversation_id: uuid.UUID | None,
    session_id: uuid.UUID | None,
    memory_update: dict[str, Any],
) -> list[MemoryCandidate]:
    if repo_id is None:
        return []
    specs = [
        ("decision", memory_update.get("decisions") or []),
        ("fact", memory_update.get("facts") or []),
        ("task", memory_update.get("tasks") or []),
        ("preference", memory_update.get("user_preferences") or []),
        ("repo_context", memory_update.get("repo_context") or []),
    ]
    created: list[MemoryCandidate] = []
    for kind, values in specs:
        for value in values:
            content = str(value or "").strip()
            if not content or _candidate_exists(db, repo_id, kind, content):
                continue
            candidate = MemoryCandidate(
                repo_id=repo_id,
                conversation_id=conversation_id,
                session_id=session_id,
                kind=kind,
                title=_candidate_title(kind, content),
                content=content,
                status="pending",
                source="session_sealer",
                meta={"memory_update_key": kind, "captured_at": datetime.now(timezone.utc).isoformat()},
            )
            db.add(candidate)
            created.append(candidate)
    if created:
        db.flush()
    return created


def list_memory_candidates(
    db: Session,
    *,
    repo_id: uuid.UUID,
    status: str | None = "pending",
    limit: int = 50,
) -> list[dict[str, Any]]:
    query = db.query(MemoryCandidate).filter(MemoryCandidate.repo_id == repo_id)
    if status and status != "all":
        query = query.filter(MemoryCandidate.status == status)
    rows = query.order_by(MemoryCandidate.created_at.desc()).limit(min(max(limit, 1), 100)).all()
    return [_candidate_payload(row) for row in rows]


# 先按仓库定位候选；已经 approved 时返回原候选，不再次制造一份重复文档。
# 新批准通过 add_document 写成 memory_note，并把候选 ID/类型等元数据附到正式文档。
# 更新 approved 状态、审阅时间和已创建文档 ID，方便后续从候选追到知识片段。
# commit=False/flush 让路由统一提交；生成向量或外部存储失败仍应按真实状态处理。
def approve_memory_candidate(db: Session, repo: Repository, candidate_id: uuid.UUID) -> tuple[MemoryCandidate, Document | None]:
    candidate = _candidate_or_none(db, repo.id, candidate_id)
    if candidate is None:
        raise ValueError("candidate_not_found")
    if candidate.status == "approved":
        return candidate, None
    doc = add_document(
        db,
        repo.id,
        "memory_note",
        uuid.uuid4(),
        candidate.title,
        candidate.content,
        {
            "display_title": candidate.title,
            "candidate_id": str(candidate.id),
            "candidate_kind": candidate.kind,
            "source": "approved_memory_candidate",
            **(candidate.meta or {}),
        },
        commit=False,
    )
    candidate.status = "approved"
    candidate.reviewed_at = datetime.now(timezone.utc)
    candidate.meta = {**(candidate.meta or {}), "approved_document_id": str(doc.id)}
    db.flush()
    return candidate, doc


# 将指定仓库候选标记 rejected 并记录审阅时间，不删除原始对话或封存现场。
# 拒绝意味着这条提取不进入正式长期知识，不代表它从所有历史来源里被抹除。
# 审阅结果供界面筛选和后续治理，真实原文仍可按会话范围核对。
def reject_memory_candidate(db: Session, repo: Repository, candidate_id: uuid.UUID) -> MemoryCandidate:
    candidate = _candidate_or_none(db, repo.id, candidate_id)
    if candidate is None:
        raise ValueError("candidate_not_found")
    candidate.status = "rejected"
    candidate.reviewed_at = datetime.now(timezone.utc)
    db.flush()
    return candidate


def _candidate_or_none(db: Session, repo_id: uuid.UUID, candidate_id: uuid.UUID) -> MemoryCandidate | None:
    return (
        db.query(MemoryCandidate)
        .filter(MemoryCandidate.repo_id == repo_id, MemoryCandidate.id == candidate_id)
        .one_or_none()
    )


def _candidate_exists(db: Session, repo_id: uuid.UUID, kind: str, content: str) -> bool:
    return (
        db.query(MemoryCandidate.id)
        .filter(
            MemoryCandidate.repo_id == repo_id,
            MemoryCandidate.kind == kind,
            MemoryCandidate.content == content,
            MemoryCandidate.status.in_(["pending", "approved"]),
        )
        .first()
        is not None
    )


def _candidate_title(kind: str, content: str) -> str:
    labels = {
        "decision": "决策",
        "fact": "事实",
        "task": "任务",
        "preference": "偏好",
        "repo_context": "项目上下文",
    }
    return f"{labels.get(kind, kind)}：{_clip(content, 80)}"


def _recall_result_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(item.get("title") or ""),
        "source_type": str(item.get("source_type") or ""),
        "source_id": str(item.get("source_id") or ""),
        "score": item.get("score"),
        "snippet": _clip(str(item.get("snippet") or ""), 260),
        "metadata": item.get("metadata") or {},
        "retrieval": item.get("retrieval") or {},
    }


def _recall_event_payload(row: RecallEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "repo_id": row.repo_id,
        "conversation_id": row.conversation_id,
        "tool_name": row.tool_name,
        "query": row.query,
        "scope": row.scope,
        "mode": row.mode,
        "status": row.status,
        "result_count": row.result_count,
        "results": row.results_json or [],
        "citations": row.citations_json or [],
        "metadata": row.meta or {},
        "created_at": row.created_at,
    }


def _candidate_payload(row: MemoryCandidate) -> dict[str, Any]:
    return {
        "id": row.id,
        "repo_id": row.repo_id,
        "conversation_id": row.conversation_id,
        "session_id": row.session_id,
        "kind": row.kind,
        "title": row.title,
        "content": row.content,
        "status": row.status,
        "source": row.source,
        "metadata": row.meta or {},
        "created_at": row.created_at,
        "reviewed_at": row.reviewed_at,
    }


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."
