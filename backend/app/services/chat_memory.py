# 会话存储与上下文装配：持久 transcript、模型活动上下文、封存摘要和检索证据各有职责。
# 核心区别：timeline 供界面与精确回忆读取完整记录；recent 只返回压缩边界之后的活动投影。
# 一次对话从 MessageStore 写消息，ContextAssembler 装输入，ProgressiveContextManager 控预算，SessionSealer 封存。
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AgentRun,
    ChatMessage,
    ChatSession,
    ChatSession as ChatSessionModel,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)
from app.services.context_compression import (
    COMPACT_BOUNDARY_SUBTYPE,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    adjust_keep_start_to_preserve_api_invariants,
    build_compaction_summary_payload,
    calculate_context_pressure,
    clip_to_token_budget,
    compress_messages,
    default_context_budget,
    dumps_compact,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    extract_relevant_snippet,
    fallback_memory_update,
    fallback_compaction_summary,
    format_compaction_summary,
    is_compact_boundary_message,
    messages_after_compact_boundary,
    memory_document_text,
    merge_memory,
    microcompact_messages,
    render_structured_memory,
    structured_memory_from_model,
)
from app.services.llm.client import LLMClient
from app.services.memory_hub import capture_memory_candidates, record_recall_event
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES
from app.services.rag.vector_store import MilvusUnavailableError


def _repo_filter(model: Any, repo_id: uuid.UUID | None) -> Any:
    return model.repo_id.is_(None) if repo_id is None else model.repo_id == repo_id


def _conversation_filter(model: Any, conversation_id: uuid.UUID | None) -> Any:
    return model.conversation_id.is_(None) if conversation_id is None else model.conversation_id == conversation_id


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text.lower())


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


class MessageStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    # 创建 ChatMessage 并 flush，使本次工作单元能立即拿到消息 UUID。
    # flush 把待写变更送到数据库事务中，但不是 commit；上层仍可将整轮记录一起提交或回滚。
    # tool_calls/meta 缺失时用空容器，避免后续消费者对 None 额外分支。
    # repo_id 与 conversation_id 同时保存，后续读取历史时要同时约束它们。
    def append(
        self,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        role: str,
        content: str,
        route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            route=route,
            tool_calls=tool_calls or [],
            meta=meta or {},
        )
        # flush 让新记录取得默认值/主键并参与当前事务；真正持久提交由上层 commit 决定。
        self.db.add(message)
        self.db.flush()
        return message

    # 取“模型当前应该看见的上下文投影”，不是界面完整聊天历史。
    # 没有压缩边界时，直接取 timeline；有边界时，用边界摘要 + 引用保留的原消息 + 边界后新消息拼接。
    # preserved_message_ids 先转 UUID，再在同仓库/会话范围回查，按 ID 清单顺序恢复。
    # 拼接时排除已经引用的消息与旧版兼容副本，防止同一句话在上下文里出现两遍。
    # 最后保留尾部 limit 条，控制投影大小；原始数据库消息没有因此被删除。
    # 例如旧消息 1..100 被摘要后，可保留原消息 93..100 的引用，再接新消息 101，而非复制 93..100。
    def recent(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 20) -> list[ChatMessage]:
        """Return the active context projection, not the complete transcript.

        New compact boundaries keep references to the original recent messages
        instead of copying them.  Older databases can still contain
        ``preserved_after_compact`` copies, so this method understands both
        layouts during the migration period.
        """
        boundary = self.last_compact_boundary(repo_id, conversation_id)
        if boundary is None:
            return self.timeline(repo_id, conversation_id, limit=limit, include_system=True)

        boundary_meta = boundary.meta or {}
        preserved_ids = [
            value
            for value in boundary_meta.get("preserved_message_ids") or []
            if value
        ]
        preserved_rows: list[ChatMessage] = []
        if preserved_ids:
            parsed_ids: list[uuid.UUID] = []
            # 摘要边界保存的原消息 ID 需要逐个解析；无效 ID 单独跳过，不让一条坏元数据阻止整段历史恢复。
            for value in preserved_ids:
                try:
                    parsed_ids.append(uuid.UUID(str(value)))
                except ValueError:
                    continue
            if parsed_ids:
                by_id = {
                    row.id: row
                    for row in self.db.query(ChatMessage)
                    .filter(
                        _repo_filter(ChatMessage, repo_id),
                        _conversation_filter(ChatMessage, conversation_id),
                        ChatMessage.id.in_(parsed_ids),
                    )
                    .all()
                }
                preserved_rows = [by_id[item] for item in parsed_ids if item in by_id]

        rows_after_boundary = (
            self.db.query(ChatMessage)
            .filter(
                _repo_filter(ChatMessage, repo_id),
                _conversation_filter(ChatMessage, conversation_id),
                ChatMessage.created_at >= boundary.created_at,
            )
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .all()
        )
        # 摘要先提供早期背景，紧接保留的原消息细节，再接边界后的新消息，形成连贯活动上下文。
        projected = [boundary, *preserved_rows]
        projected.extend(
            row
            for row in rows_after_boundary
            if row.id != boundary.id
            and str(row.id) not in preserved_ids
            and not (
                preserved_ids
                and (row.meta or {}).get("subtype") == "preserved_after_compact"
            )
        )
        return projected[-max(1, limit) :]

    # 完整持久 transcript 的读取路径，界面回看和精确回忆应使用它。
    # 先按仓库/会话限定范围，再按 created_at、id 形成稳定全序，避免同一时间戳的消息分页重复或遗漏。
    # before/after 游标也使用“时间 + ID”的联合比较，与排序规则保持一致。
    # 隐藏旧 preserved_after_compact 兼容副本，但保留用户/assistant 原始消息；include_system 控制是否包含系统边界。
    # 向后翻页取较早区间末尾，after 单向前进则取新区间开头；两种方向不能机械用同一段切片。
    # limit=None 表示调用方明确要全部持久记录，不执行尾部数量截断。
    def timeline(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        *,
        limit: int | None = 500,
        include_system: bool = False,
        before_message_id: uuid.UUID | None = None,
        after_message_id: uuid.UUID | None = None,
    ) -> list[ChatMessage]:
        """Return the durable transcript while hiding compatibility copies.

        Compact boundaries change the model's active projection; they do not
        erase the transcript.  This method is the authoritative read path for
        UI history and exact-memory tools.
        """
        query = self.db.query(ChatMessage).filter(
            _repo_filter(ChatMessage, repo_id),
            _conversation_filter(ChatMessage, conversation_id),
        )
        if not include_system:
            query = query.filter(ChatMessage.role != "system")

        if before_message_id:
            before = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == before_message_id,
                )
                .one_or_none()
            )
            if before and before.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at < before.created_at,
                        and_(
                            ChatMessage.created_at == before.created_at,
                            ChatMessage.id < before.id,
                        ),
                    )
                )
        if after_message_id:
            after = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == after_message_id,
                )
                .one_or_none()
            )
            if after and after.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at > after.created_at,
                        and_(
                            ChatMessage.created_at == after.created_at,
                            ChatMessage.id > after.id,
                        ),
                    )
                )

        # 时间相同的消息用 UUID 再排序，游标也使用相同次序，确保翻页有稳定边界。
        rows = query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()
        durable = [
            row
            for row in rows
            if (row.meta or {}).get("subtype") != "preserved_after_compact"
        ]
        if limit is None:
            return durable
        if after_message_id and not before_message_id:
            return durable[: max(1, limit)]
        return durable[-max(1, limit) :]

    def active_for_context(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        limit: int = 1000,
    ) -> list[ChatMessage]:
        return self.recent(repo_id, conversation_id, limit=limit)

    def context_projection(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        rows: list[ChatMessage],
    ) -> list[dict[str, Any]]:
        """从已保存的工具事件生成本轮历史视图，不回写或复制数据库原始消息。"""
        # rows 必须先经过 recent 的摘要边界和近期窗口筛选，不能从所有历史会话重新灌入工具结果。
        assistant_ids = [row.id for row in rows if row.role == "assistant"]
        sessions: dict[uuid.UUID, ChatSession] = {}
        if assistant_ids:
            for session in (
                self.db.query(ChatSession)
                .filter(
                    _repo_filter(ChatSession, repo_id),
                    _conversation_filter(ChatSession, conversation_id),
                    ChatSession.assistant_message_id.in_(assistant_ids),
                )
                .order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
                .all()
            ):
                sessions.setdefault(session.assistant_message_id, session)

        # 同一事件同时出现在 agent_events 和 tool_use_results 时只选一个索引；原生工具行也不能重复注入。
        projected: list[dict[str, Any]] = []
        native_result_ids: set[str] = set()
        for row in rows:
            meta = dict(row.meta or {})
            if row.role == "user":
                native_result_ids.clear()
            if row.role in {"tool", "tool_result"}:
                call_id = meta.get("tool_use_id") or meta.get("tool_call_id")
                if call_id:
                    native_result_ids.add(str(call_id))
                meta.setdefault("recovery", {
                    "tool": "devflow_get_thread_context", "message_id": str(row.id),
                })
            if row.role == "assistant" and meta.get("subtype") != "tool_result":
                session = sessions.get(row.id)
                transcript = (session.transcript_json or {}) if session else {}
                events = transcript.get("tool_use_results") or transcript.get("agent_events") or meta.get("agent_events") or []
                events = [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
                uses = {
                    str(event["id"]): event
                    for event in events
                    if event.get("type") == "tool_use" and event.get("id")
                }
                seen = set(native_result_ids)
                for index, event in enumerate(events):
                    if event.get("type") != "tool_result" or event.get("content") is None:
                        continue
                    call_id = str(event.get("tool_use_id") or event.get("tool_call_id") or f"{row.id}:{index}")
                    if call_id in seen:
                        continue
                    seen.add(call_id)
                    use = uses.get(call_id, {})
                    tool_name = str(event.get("tool_name") or use.get("tool_name") or "unknown_tool")
                    arguments = use.get("arguments") or {}
                    event_meta = event.get("meta") or event.get("metadata") or {}
                    event_meta = event_meta if isinstance(event_meta, dict) else {}
                    # 有封存会话时保留精确回读入口；旧元数据模式则保留原工具及参数，便于重新取数。
                    recovery = (
                        {"tool": "devflow_read_session_events", "session_id": str(session.id), "include_transcript": True}
                        if session else {"tool": tool_name, "arguments": arguments}
                    )
                    projected.append({
                        "role": "tool_result",
                        "route": tool_name,
                        "content": dumps_compact({"arguments": arguments, "result": event["content"], "status": event.get("status")}),
                        "meta": {
                            "subtype": "tool_result",
                            "tool_use_id": call_id,
                            "source_message_id": str(row.id),
                            "regenerable": event.get("regenerable") is not False
                            and event_meta.get("regenerable") is not False
                            and event.get("status") != "error",
                            "recovery": recovery,
                        },
                    })
            projected.append({
                "id": str(row.id),
                "role": row.role,
                "content": row.content,
                "route": row.route,
                "tool_calls": row.tool_calls or [],
                "meta": meta,
            })
        return projected

    def all_for_context(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 200) -> list[ChatMessage]:
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def last_compact_boundary(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ChatMessage | None:
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id), ChatMessage.role == "system")
            .order_by(ChatMessage.created_at.desc())
            .limit(50)
            .all()
        )
        # 候选按新到旧读取，逐条识别系统摘要边界，找到第一条即可停止，不能误把普通系统消息当边界。
        for row in rows:
            if is_compact_boundary_message({"role": row.role, "meta": row.meta or {}}):
                return row
        return None


class EvidenceStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    # 为本轮装配准备统一证据投影，将不同业务对象转成相同的标题/内容/元数据形状。
    # 来源仍保留各自标识，后续召回可以说明来自 Issue、PR、CI、文档或会话。
    # 重建范围与当前仓库、会话有关，不能把所有线程的讨论无差别混进当前上下文。
    # 它服务于当前证据查询，与“所有数据都要向量化”不是同一个要求；来源策略仍需分别遵守。
    def rebuild_repo(self, repo: Repository | None, conversation: Conversation | None = None) -> None:
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        if conversation_id:
            self.db.query(EvidenceItem).filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            ).delete(synchronize_session=False)
        else:
            self.db.query(EvidenceItem).filter(_repo_filter(EvidenceItem, repo_id)).delete(synchronize_session=False)

        if repo:
            self._add(
                repo_id,
                None,
                "repository",
                str(repo.id),
                repo.full_name,
                "\n".join(filter(None, [repo.description or "", f"默认分支：{repo.default_branch or '未知'}"])),
                {"anchor": f"repo:{repo.full_name}"},
            )
            # 每个 Issue 建立独立证据项，保留来源身份；不能把整个仓库的问题拼成一份失去定位的大文本。
            for issue in self.db.query(Issue).filter(Issue.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "issue",
                    str(issue.id),
                    f"Issue #{issue.number}: {issue.title}",
                    "\n".join(filter(None, [issue.title, issue.body or "", f"state: {issue.state}", f"labels: {', '.join(issue.labels or [])}"])),
                    {"number": issue.number, "state": issue.state},
                )
            # PR 描述与 Issue 分开建项，便于检索后按来源类型回查，而不是混用两个表的 ID。
            for pr in self.db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "pull_request",
                    str(pr.id),
                    f"PR #{pr.number}: {pr.title}",
                    "\n".join(filter(None, [pr.title, pr.body or "", f"state: {pr.state}", f"base: {pr.base_branch}", f"head: {pr.head_branch}"])),
                    {"number": pr.number, "state": pr.state},
                )
            # 逐个检查流水线，仅收录有失败日志的记录，并先脱敏再截取，避免正常记录和敏感日志进入证据。
            for run in self.db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all():
                if run.conclusion != "failure" or not (run.logs_text or "").strip():
                    continue
                sanitized_logs = sanitize_ci_log(run.logs_text or "")
                safe_logs = sanitized_logs[:CI_LOG_CHUNK_CHARS]
                self._add(
                    repo.id,
                    None,
                    "workflow_run",
                    str(run.id),
                    f"CI: {run.name}",
                    "\n".join(filter(None, [run.name, run.status, run.conclusion or "", safe_logs])),
                    {
                        "status": run.status,
                        "conclusion": run.conclusion,
                        "logs_truncated": len(sanitized_logs) > CI_LOG_CHUNK_CHARS,
                        "secrets_redacted": True,
                    },
                )
            # 关系库中符合 RAG 来源策略的文档也转成证据项；这只是证据存储，不把所有源码都重新向量化。
            for doc in (
                self.db.query(Document)
                .filter(Document.repo_id == repo.id, Document.source_type.in_(RAG_SOURCE_TYPES))
                .all()
            ):
                self._add(repo.id, None, doc.source_type, str(doc.source_id), doc.title, doc.content, doc.meta or {})

        memories = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).all()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).all()
        )
        # 把当前范围内的记忆摘要分别建项，保留会话边界，不把其他会话私有记忆随意并入。
        for memory in memories:
            self._add(
                repo_id,
                conversation_id,
                "memory",
                str(memory.id),
                "Thread memory",
                memory_document_text(memory),
                {"sessions_incorporated": memory.sessions_incorporated},
            )
        session_query = self.db.query(ChatSession).filter(_repo_filter(ChatSession, repo_id))
        run_query = self.db.query(AgentRun).filter(_repo_filter(AgentRun, repo_id))
        if conversation_id:
            session_query = session_query.filter(ChatSession.conversation_id == conversation_id)
            run_query = run_query.filter(AgentRun.conversation_id == conversation_id)
        # 只投影近期封存摘要，给证据检索提供跨轮线索；完整原文仍从会话工具读取。
        for session in session_query.order_by(ChatSession.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "session", str(session.id), f"Session digest: {session.route}", session.digest, {"route": session.route})
        # 近期运行的问题与结果作为独立可定位记录，不与会话摘要共用来源身份。
        for run in run_query.order_by(AgentRun.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "agent_run", str(run.id), f"Agent run: {run.route}", f"{run.user_message}\n{run.final_answer}", {"route": run.route})
        self.db.flush()

    # 围绕当前问题查找可用于本轮的证据，同时保持仓库与会话范围。
    # 返回的是适合装配的片段结构，标题、来源、分数和元数据便于下游压缩与追踪。
    # limit 控制证据数量而不是每段文本长度，装配时还要再分配 token 预算。
    # 命中只代表候选相关；实时研发状态或精确代码结论仍应回到数据库/checkout 核对。
    def search(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        results: list[dict[str, Any]] = []  # 本次检索的最终证据，先装 RAG 命中，再用本地词面证据补足
        seen: set[tuple[str, str]] = set()  # 已返回的来源类型和来源 ID 组合，防止两条检索路径重复占名额
        if repo_id:
            try:
                rag_results = search_similar_documents(
                    self.db,
                    repo_id,
                    query,
                    source_type=None,
                    limit=max(limit, 8),
                )
            except MilvusUnavailableError:
                rag_results = []
            # 逐条转换 RAG 候选为统一的证据形状，并登记去重身份；达到目标条数即可提前返回。
            for item in rag_results:
                key = (str(item.get("source_type") or ""), str(item.get("source_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "title": item.get("title") or "",
                        "source_type": item.get("source_type") or "",
                        "source_id": item.get("source_id") or "",
                        "snippet": item.get("snippet") or "",
                        "score": item.get("score") or 0,
                        "metadata": item.get("metadata") or {},
                        "retrieval": item.get("retrieval") or {},
                    }
                )
                if len(results) >= limit:
                    return results
        candidates = (
            self.db.query(EvidenceItem)
            .filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            )
            .order_by(EvidenceItem.created_at.desc())
            .limit(500)
            .all()
        )
        ranked: list[tuple[int, EvidenceItem]] = []  # 暂存关键词得分与证据对象，评分完成后统一排序
        # 本地候选需要逐条计分，而不是直接把最近的 500 条都交给模型；已在 RAG 返回的来源跳过。
        for item in candidates:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            haystack = f"{item.title}\n{item.content}".lower()
            # 这里是词面命中次数之和，只表示排序相关性，不能与向量分或事实置信度等同。
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                ranked.append((score, item))
        # 优先命中更多查询词的记录，同分时较新的在前，结果顺序与筛选规则保持可解释。
        ranked.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        # 按排名逐项补入，边补边检查去重和 limit；无需先为所有低排名项生成片段。
        for score, item in ranked:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "snippet": extract_relevant_snippet(query, item.content, max_tokens=90),
                    "score": score,
                    "metadata": item.meta or {},
                    "retrieval": {"sources": ["evidence_item_keyword"]},
                }
            )
            if len(results) >= limit:
                break
        return results

    def _add(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, source_type: str, source_id: str, title: str, content: str, meta: dict[str, Any]) -> None:
        if not content.strip():
            return
        self.db.add(EvidenceItem(repo_id=repo_id, conversation_id=conversation_id, source_type=source_type, source_id=source_id, title=title, content=content, meta=meta))


@dataclass
class AssembledContext:
    system_context: str  # 项目基础信息、记忆和证据合成的背景字符串
    history_messages: list[dict[str, str]]  # 单独返回的近期消息，避免在背景中重复嵌入
    evidence: list[dict[str, Any]]  # 本轮原问题召回的候选证据
    compression_stats: dict[str, Any] = field(default_factory=dict)  # 预算、覆盖、压缩与压力等诊断


class ContextAssembler:
    def __init__(self, db: Session) -> None:
        self.db = db

    # 输入是当前仓库、会话和新问题，输出分开的 system_context、history_messages、evidence 与统计。
    # 先读取最近压缩边界、结构化记忆和上一轮封存摘要，再按原始问题检索相关证据。
    # 最近历史取比展示上限更宽的窗口，计算压力后可先微压缩，再按 recent_tokens 留输入片段。
    # 若新问题已作为最后一条 user 消息存入历史，就不再重复计入 pending_user_tokens。
    # build_context 只装稳定系统信息、记忆与证据，历史通过 history_messages 单独返回，避免双份注入。
    # 最后重新估算装配压力，记录历史是否省略/截断/跨摘要边界；complete_and_exact 只在条件全部满足时成立。
    # 这些覆盖信息供回忆策略使用：模型看到一部分历史，不能声称自己拥有完整原文。
    def assemble(self, repo: Repository | None, conversation: Conversation | None, current_message: str, limit: int = 16) -> AssembledContext:
        budget = default_context_budget()
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        message_store = MessageStore(self.db)
        boundary = message_store.last_compact_boundary(repo_id, conversation_id)
        memory = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        )
        latest_session = (
            self.db.query(ChatSession)
            .filter(
                _repo_filter(ChatSession, repo_id),
                _conversation_filter(ChatSession, conversation_id),
                ChatSession.status.in_(["sealed", "sealed_with_warnings"]),
            )
            .order_by(ChatSession.created_at.desc())
            .first()
        )
        evidence = EvidenceStore(self.db).search(repo_id, conversation_id, current_message, limit=8)
        history_fetch_limit = max(limit, 30)
        recent = message_store.recent(repo_id, conversation_id, limit=history_fetch_limit)
        # 除用户与助手正文外，也展开关联会话中保存的工具结果，交给微压缩后再进入历史预算。
        raw_history = [
            item for item in message_store.context_projection(repo_id, conversation_id, recent)
            if item["role"] in {"user", "assistant", "tool", "tool_use", "tool_result"}
        ]
        micro_stats: dict[str, Any] | None = None
        raw_history_tokens = sum(
            estimate_tokens(str(item.get("role") or ""))
            + estimate_tokens(str(item.get("content") or ""))
            + (
                estimate_tokens(dumps_compact(item.get("tool_calls")))
                if item.get("tool_calls")
                else 0
            )
            for item in raw_history
        )
        current_message_in_history = bool(
            raw_history
            and raw_history[-1].get("role") == "user"
            and " ".join(str(raw_history[-1].get("content") or "").split())
            == " ".join(str(current_message or "").split())
        )
        # 路由可能已先写入本轮 user 消息；计两遍会虚增压力，导致过早压缩。
        pending_user_tokens = 0 if current_message_in_history else estimate_tokens(current_message)
        # ensure_headroom 在本轮 user 入库前执行；待写入的问题也占近期保护区的一条，保证前后两次装配一致。
        micro_keep_recent = max(
            0, settings.context_microcompact_keep_recent_messages - int(bool(pending_user_tokens)),
        )
        raw_history_pressure = calculate_context_pressure(raw_history_tokens + pending_user_tokens)
        history_source = raw_history
        if raw_history_pressure.should_warn:
            history_source, micro_stats = microcompact_messages(raw_history, current_message, keep_recent=micro_keep_recent)
        # 此时只压缩传给模型的列表；数据库里原消息正文与 created_at 不做修改。
        history, history_stats = compress_messages(history_source, budget.recent_tokens)

        # 把系统、记忆和证据放入各自预算，返回一个服务端背景字符串。
        # 记忆包含摘要边界、可恢复工作状态、结构化记忆和上一轮摘要，按可用空间裁剪。
        # 证据预算先按条目数量分摊，再对整体证据区设上限，避免少数长日志占满上下文。
        # next_history 虽是函数参数，但当前正文不在这里再次嵌入历史，历史由 assemble 单独返回。
        # 稳定区与易变区分开，也更容易定位到底是哪一部分导致上下文压力。
        def build_context(next_history: list[dict[str, str]], next_evidence: list[dict[str, Any]]) -> str:
            system_sections: list[str] = []
            if repo:
                system_sections.append(f"[Current Repository]\n{repo.full_name}\n[/Current Repository]")
            if conversation:
                system_sections.append(f"[Current Conversation]\n{conversation.title}\n[/Current Conversation]")

            sections: list[str] = []
            stable_system = clip_to_token_budget("\n\n".join(system_sections), budget.system_tokens)
            if stable_system:
                sections.append(stable_system)

            memory_sections: list[str] = []
            if boundary and boundary.content:
                transcript_ref = (boundary.meta or {}).get("transcript_ref") or {}
                rehydration = (boundary.meta or {}).get("rehydration") or {}
                recovery_hint = (
                    "\nExact transcript recovery: call devflow_get_thread_context with this conversation"
                    if transcript_ref
                    else ""
                )
                if rehydration:
                    memory_sections.append(
                        "[Rehydrated Working State]\n"
                        + dumps_compact(rehydration)
                        + "\n[/Rehydrated Working State]"
                    )
                memory_sections.append(
                    f"[Compaction Boundary]\n{boundary.content}{recovery_hint}\n[/Compaction Boundary]"
                )
            if memory and memory_document_text(memory).strip():
                memory_text = render_structured_memory(memory, max_tokens=max(256, budget.memory_tokens // 2))
                memory_sections.append(f"[Thread Memory: {memory.sessions_incorporated} sessions]\n{memory_text}\n[/Thread Memory]")
            if latest_session and latest_session.digest:
                memory_sections.append(f"[Previous Session Digest]\n{latest_session.digest}\n[/Previous Session Digest]")
            memory_context = clip_to_token_budget("\n\n".join(memory_sections), budget.memory_tokens)
            if memory_context:
                sections.append(memory_context)
            # 只有命中的证据才加入 Related Evidence；没有证据时不构造一个看似有资料的空证明区。
            if next_evidence:
                evidence_budget = max(120, budget.evidence_tokens // max(len(next_evidence), 1))
                lines = ["[Related Evidence]"]
                # 每条证据先按分摊额度裁剪，再把整个证据区按总预算裁剪；两层限制防止单条或总量失控。
                for item in next_evidence:
                    snippet = clip_to_token_budget(item["snippet"], evidence_budget)
                    lines.append(f"- [{item['source_type']}] {item['title']}: {snippet}")
                lines.append("[/Related Evidence]")
                sections.append(clip_to_token_budget("\n".join(lines), budget.evidence_tokens))
            return "\n\n".join(sections)

        system_context = build_context(history, evidence)
        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        initial_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        initial_pressure = calculate_context_pressure(initial_prompt_tokens)
        if initial_pressure.should_warn and micro_stats is None:
            micro_raw_history, micro_stats = microcompact_messages(raw_history, current_message, keep_recent=micro_keep_recent)
            history, history_stats = compress_messages(micro_raw_history, budget.recent_tokens)
            system_context = build_context(history, evidence)

        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        assembled_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        final_pressure = calculate_context_pressure(assembled_prompt_tokens)
        microcompacted_messages = int((micro_stats or {}).get("compacted_messages") or 0)
        history_coverage = {
            "source_messages": len(raw_history),
            "prompt_messages": len(history),
            "omitted_messages": int(history_stats.get("omitted_messages") or 0),
            "truncated_messages": int(history_stats.get("truncated_messages") or 0),
            "exact_suffix_messages": int(history_stats.get("exact_suffix_messages") or 0),
            "source_limit_reached": len(recent) >= history_fetch_limit,
            "has_compaction_boundary": boundary is not None,
            "microcompacted_messages": microcompacted_messages,
        }
        # “完整且精确”要求所有负面条件均为假，不能只因还剩几条近期消息就认为没有截断。
        history_coverage["complete_and_exact"] = not any(
            [
                history_coverage["omitted_messages"],
                history_coverage["truncated_messages"],
                history_coverage["source_limit_reached"],
                history_coverage["has_compaction_boundary"],
                history_coverage["microcompacted_messages"],
            ]
        )

        return AssembledContext(
            system_context=system_context,
            history_messages=history,
            evidence=evidence,
            compression_stats={
                "budget": budget.__dict__,
                "history": history_stats,
                "history_coverage": history_coverage,
                "microcompact": micro_stats,
                "pressure": final_pressure.as_dict(),
                "initial_pressure": initial_pressure.as_dict(),
                "raw_history_pressure": raw_history_pressure.as_dict(),
                "compact_boundary": {
                    "id": str(boundary.id) if boundary else None,
                    "created_at": boundary.created_at.isoformat() if boundary and boundary.created_at else None,
                    "mode": (boundary.meta or {}).get("mode") if boundary else None,
                },
                "system_context_estimated_tokens": estimate_tokens(system_context),
                "history_estimated_tokens": history_prompt_tokens,
                "pending_user_estimated_tokens": pending_user_tokens,
                "assembled_prompt_estimated_tokens": assembled_prompt_tokens,
                "prompt_breakdown": {
                    "system_memory_evidence": estimate_tokens(system_context),
                    "recent_history": history_prompt_tokens,
                    "pending_user": pending_user_tokens,
                    "tool_observation_budget": budget.tool_observation_tokens,
                    "reserved_response": budget.reserved_response_tokens,
                },
                "evidence_items": len(evidence),
                "mode": "progressive_budgeted_context",
            },
        )


class ProgressiveContextManager:
    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    # 新一轮发问前，分别估算原始活动段和最终装配输入的压力，决定是否需要正式摘要。
    # 如果微压缩已让实际提示回到 normal/warning，即使原始段仍很长也可以不立即正式压缩。
    # 关闭自动压缩、当前压力正常、历史太短等情况都有明确 reason，便于前端诊断。
    # 连续失败次数达到上限后打开熔断，避免每次请求都再次失败、不断消耗摘要调用。
    # 正式压缩失败会回滚本次数据库工作并写失败记录；返回统计让上层决定继续响应方式。
    # 它只准备上下文空间，不执行业务分析，也不替路由提交整轮对话事务。
    async def ensure_headroom(self, repo: Repository | None, conversation: Conversation | None, current_message: str) -> dict[str, Any]:
        if repo is None or conversation is None:
            return {"attempted": False, "reason": "no_repo_or_conversation"}
        assembled = ContextAssembler(self.db).assemble(repo, conversation, current_message)
        assembled_pressure = assembled.compression_stats.get("pressure") or {}
        micro_stats = assembled.compression_stats.get("microcompact") or {}
        raw_segment_tokens = self._raw_segment_tokens(repo.id, conversation.id) + estimate_tokens(current_message)
        assembled_tokens = int(
            assembled.compression_stats.get("assembled_prompt_estimated_tokens")
            or assembled.compression_stats.get("system_context_estimated_tokens")
            or 0
        )
        raw_pressure_state = calculate_context_pressure(raw_segment_tokens)
        assembled_pressure_state = calculate_context_pressure(assembled_tokens)
        # 同时关注原始活动段和已装配输入，取更大的压力；微压缩已经省掉的空间也另行记录。
        pressure_state = calculate_context_pressure(max(raw_segment_tokens, assembled_tokens))
        pressure = {
            **pressure_state.as_dict(),
            "raw_segment_estimated_tokens": raw_segment_tokens,
            "assembled_context_estimated_tokens": assembled_tokens,
            "raw_stage": raw_pressure_state.stage,
            "assembled_stage": assembled_pressure.get("stage") or assembled_pressure_state.stage,
            "microcompact_estimated_tokens_saved": micro_stats.get("estimated_tokens_saved", 0),
            "microcompact_compacted_messages": micro_stats.get("compacted_messages", 0),
        }
        raw_stage = raw_pressure_state.stage
        prompt_stage = str(pressure.get("assembled_stage") or assembled_pressure_state.stage)
        stage = str(pressure.get("stage") or "normal")
        # 即使压力很高，关闭自动摘要时也不能擅自写摘要边界；返回压力信息让上层明确展示。
        if not settings.context_auto_compact_enabled:
            return {"attempted": False, "reason": "auto_compact_disabled", "pressure": pressure}
        if (
            prompt_stage in {"normal", "warning"}
            and raw_stage in {"auto_compact", "aggressive", "manual_path"}
            and int(micro_stats.get("compacted_messages") or 0) > 0
        ):
            return {
                "attempted": False,
                "reason": "microcompact_sufficient",
                "pressure": pressure,
                "microcompact": micro_stats,
            }
        # 尚未达到正式摘要阶段时只报告压力，避免每轮对话都触发一次收费模型总结。
        if stage in {"normal", "warning"}:
            return {"attempted": False, "reason": stage, "pressure": pressure}

        # 统计的是当前边界后的连续失败，不是所有历史失败总和；成功恢复后不应永久熔断。
        failures = self._consecutive_failures(repo.id, conversation.id)
        # 连续失败触发熔断，防止每次追问都重复付出摘要成本却没有有效释放窗口。
        if failures >= settings.context_max_compaction_failures:
            return {
                "attempted": False,
                "reason": "circuit_breaker_open",
                "consecutive_failures": failures,
                "pressure": pressure,
            }

        try:
            result = await self._compact(repo, conversation, current_message, pressure, stage)
            return {"attempted": True, **result, "pressure": pressure, "consecutive_failures": 0}
        except Exception as exc:  # pragma: no cover - defensive circuit breaker path
            self.db.rollback()
            failure = self._record_failure(repo.id, conversation.id, current_message, stage, exc)
            self.db.flush()
            return {
                "attempted": True,
                "compacted": False,
                "reason": "compaction_failed",
                "failure_id": str(failure.id),
                "error": str(exc),
                "pressure": pressure,
                "consecutive_failures": failures + 1,
            }

    # 只处理最近压缩边界之后的活动段，先选择用户/助手/工具等可摘要消息。
    # 按当前压力准备若干策略：默认、更加紧凑、极限路径；每种有不同的保留消息数和摘要预算。
    # 先算要保留哪些原消息，其余作为摘要输入，同时抽取后续恢复任务需要的工作状态。
    # 每次生成摘要后估算“摘要 + 保留消息 + 新问题 + 恢复状态”的总压力；够用就停止更激进尝试。
    # 选定策略后插入边界，记录被摘要范围、原消息引用、事实保留报告和策略诊断。
    # 原始 transcript 不在此删除；精确追问仍可通过记忆工具跨边界取回原文。
    # 若没有足够旧消息可摘要，返回 not_enough_messages/nothing_to_summarize，不制造空边界。
    async def _compact(
        self,
        repo: Repository,
        conversation: Conversation,
        current_message: str,
        pressure: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        store = MessageStore(self.db)
        all_rows = store.active_for_context(repo.id, conversation.id, limit=500)
        all_messages = [self._row_to_message(row) for row in all_rows]
        segment_messages = messages_after_compact_boundary(all_messages, include_boundary=False)
        # 先用活动段消息身份筛 ORM 行，确保摘要只覆盖最新边界后的真实记录。
        message_ids = {str(item.get("id")) for item in segment_messages if item.get("id")}
        segment_rows = [row for row in all_rows if str(row.id) in message_ids]
        candidate_rows = [
            row
            for row in segment_rows
            if row.role in {"user", "assistant", "tool", "tool_use", "tool_result"}
        ]
        # 压力越高可尝试更紧策略；保留列表与摘要预算是策略，不是固定删除历史消息的规则。
        plan_stages = [stage]
        if stage == "auto_compact":
            plan_stages.extend(["aggressive", "manual_path"])
        elif stage == "aggressive":
            plan_stages.append("manual_path")
        plans = [self._compaction_plan_for_stage(item) for item in plan_stages]
        if len(candidate_rows) <= min(int(item["keep_recent"]) for item in plans) + 1:
            return {"compacted": False, "reason": "not_enough_messages"}

        # 还没有尝试任何摘要方案，诊断列表先为空，每完成一轮方案再追加结果。
        attempts: list[dict[str, Any]] = []
        selected: tuple[
            dict[str, Any],
            list[ChatMessage],
            list[ChatMessage],
            str,
            str,
            dict[str, Any],
            int,
        ] | None = None
        # 摘要之外保留可恢复的工作状态，如定位与引用，避免只剩一段无法继续操作的自然语言总结。
        # 文件定位、工具与 Skill 信息独立保留，减少自然语言摘要漏掉“下一步到哪里继续”的风险。
        rehydration = self._rehydration_state(candidate_rows)
        # 按从宽到紧的方案顺序尝试，每轮重选原文保留范围并生成摘要；index 从 1 起便于展示第几次尝试。
        for index, plan in enumerate(plans, start=1):
            keep_rows = self._messages_to_keep(candidate_rows, keep_count=int(plan["keep_recent"]))
            # 用 ID 集合判断某条原消息是否已保留，避免把相同正文的不同消息误当成同一条。
            keep_ids = {row.id for row in keep_rows}
            # 需要精确保留的原消息不再同时放进摘要输入，避免上下文中既有原文又有重复总结。
            summarize_rows = [row for row in candidate_rows if row.id not in keep_ids]
            # 如果所有候选都需原样保留，这一方案没有可摘要内容，跳过而不是调用模型生成空摘要。
            if not summarize_rows:
                continue
            summary, mode, preservation_report = await self._summary_for(
                repo,
                conversation,
                summarize_rows,
                current_message,
                max_summary_tokens=int(plan["summary_tokens"]),
            )
            mode_prefix = str(plan.get("mode_prefix") or "")
            if mode_prefix:
                mode = f"{mode_prefix}_{mode}"
            post_compact_tokens = self._post_compact_tokens(
                summary,
                keep_rows,
                current_message=current_message,
                rehydration=rehydration,
            )
            # 方案是否够用要看完整装配后的压力，而不是只看摘要本身有多短。
            post_pressure = calculate_context_pressure(post_compact_tokens)
            attempts.append(
                {
                    "attempt": index,
                    "plan": plan,
                    "post_compact_estimated_tokens": post_compact_tokens,
                    "post_compact_stage": post_pressure.stage,
                }
            )
            # 保存本次完整候选方案，供停止循环后落库；后面的更紧方案会替换它，不会并排写多个边界。
            selected = (
                plan,
                keep_rows,
                summarize_rows,
                summary,
                mode,
                preservation_report,
                post_compact_tokens,
            )
            # 尝试更紧策略前先验算新上下文压力；达到可接受范围就停，不为追求更短而继续损失细节。
            if post_pressure.stage in {"normal", "warning"}:
                break

        # 所有方案都没有可处理内容时明确退出，避免解包一个尚未产生的结果。
        if selected is None:
            return {"compacted": False, "reason": "nothing_to_summarize"}
        plan, keep_rows, summarize_rows, summary, mode, preservation_report, post_compact_tokens = selected
        boundary = self._insert_boundary(
            repo,
            conversation,
            summary,
            mode,
            pressure,
            summarize_rows,
            keep_rows,
            preservation_report,
            rehydration,
        )
        return {
            "compacted": True,
            "mode": mode,
            "stage": stage,
            "boundary_id": str(boundary.id),
            "messages_summarized": len(summarize_rows),
            "messages_preserved": len(keep_rows),
            "post_compact_estimated_tokens": post_compact_tokens,
            "post_compact_pressure": calculate_context_pressure(post_compact_tokens).as_dict(),
            "compaction_plan": plan,
            "compaction_attempts": attempts,
            "rehydration": rehydration,
        }

    def _compaction_plan_for_stage(self, stage: str) -> dict[str, Any]:
        default_keep = max(2, settings.context_compact_keep_recent_messages)
        default_summary_tokens = max(256, settings.context_compact_summary_tokens)
        if stage == "manual_path":
            return {
                "keep_recent": max(2, min(default_keep, max(2, default_keep // 3))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.5))),
                "mode_prefix": "manual_path",
            }
        if stage == "aggressive":
            return {
                "keep_recent": max(2, min(default_keep, max(3, default_keep // 2))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.67))),
                "mode_prefix": "aggressive",
            }
        return {"keep_recent": default_keep, "summary_tokens": default_summary_tokens, "mode_prefix": ""}

    # 先按目标数量选择近期后缀，再修正起点以满足工具协议与同条供应商响应的完整性。
    # 保留 N 条是近似目标：为了不留下孤立 tool_result，实际可能需要向前多保留几条。
    # 因此不能把此方法替换为 rows[-N:]，否则看似只改数量，却可能破坏下一轮 API 输入。
    def _messages_to_keep(self, rows: list[ChatMessage], keep_count: int | None = None) -> list[ChatMessage]:
        keep = max(2, keep_count if keep_count is not None else settings.context_compact_keep_recent_messages)
        start_index = max(0, len(rows) - keep)
        start_index = adjust_keep_start_to_preserve_api_invariants(
            [self._row_to_message(row) for row in rows],
            start_index,
        )
        keep_rows = rows[start_index:]
        if (
            keep_rows
            and keep_rows[0].role == "assistant"
            and start_index > 0
            and rows[start_index - 1].role == "user"
        ):
            keep_rows = [rows[start_index - 1], *keep_rows]
        return keep_rows

    # 优先请求模型生成紧凑摘要，没有可用输出时使用确定性回退摘要。
    # 随后检查必须保留的事实/任务/定位线索，必要时补回，以减少压缩丢失关键约束。
    # 返回摘要、生成模式与事实保留报告；上层还要用摘要重新计算压力，而非只看摘要字符数。
    # 这里总结的是已有对话事实，不应把摘要模型新猜测的内容提升成用户确认过的要求。
    async def _summary_for(
        self,
        repo: Repository,
        conversation: Conversation,
        rows: list[ChatMessage],
        current_message: str,
        max_summary_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        messages = MessageStore(self.db).context_projection(repo.id, conversation.id, rows)
        memory = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).first()
        if memory and memory_document_text(memory).strip():
            summary = render_structured_memory(memory, max_tokens=max_summary_tokens)
            continuation = (
                "This conversation was compacted using structured session memory. "
                "Continue from the preserved recent messages and the current user request.\n\n"
                f"{summary}"
            )
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, continuation, max_summary_tokens)
            return summary, "session_memory", report

        payload = build_compaction_summary_payload(messages)
        llm_summary = await self.llm.chat_text(
            COMPACT_SUMMARY_SYSTEM_PROMPT,
            f"Current user request that triggered pressure:\n{current_message}\n\nConversation segment to compact:\n{payload}",
        )
        summary = format_compaction_summary(llm_summary)
        if not summary:
            summary = fallback_compaction_summary(messages, max_tokens=max_summary_tokens)
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
            return summary, "fallback_extractive", report
        summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
        return summary, "full_compaction", report

    # 新建系统类型的摘要边界，meta 中保存原消息引用、摘要范围、恢复状态和诊断。
    # 保留近期消息时记录其 ID，而不是再复制一套正文，减少重复与后续界面去重负担。
    # transcript_ref 让读取工具知道还可以回到持久历史精确查原文。
    # 此方法更新模型活动上下文的解释方式，不是给数据库历史做永久截断。
    def _insert_boundary(
        self,
        repo: Repository,
        conversation: Conversation,
        summary: str,
        mode: str,
        pressure: dict[str, Any],
        summarized_rows: list[ChatMessage],
        keep_rows: list[ChatMessage],
        preservation_report: dict[str, Any],
        rehydration: dict[str, Any],
    ) -> ChatMessage:
        now = datetime.now(timezone.utc)
        latest_source_time = max(
            (row.created_at for row in [*summarized_rows, *keep_rows] if row.created_at),
            default=None,
        )
        if latest_source_time:
            comparable_now = now if latest_source_time.tzinfo else now.replace(tzinfo=None)
            if latest_source_time >= comparable_now:
                now = latest_source_time + timedelta(microseconds=1)
        boundary = ChatMessage(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="system",
            content=summary,
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": COMPACT_BOUNDARY_SUBTYPE,
                "trigger": "auto",
                "mode": mode,
                "pressure": pressure,
                "pre_compact_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "summarized_from_message_id": str(summarized_rows[0].id) if summarized_rows else None,
                "summarized_to_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "preserved_start_message_id": str(keep_rows[0].id) if keep_rows else None,
                "preserved_message_ids": [str(row.id) for row in keep_rows],
                "transcript_ref": {
                    "repo_id": str(repo.id),
                    "conversation_id": str(conversation.id),
                    "tool": "devflow_get_thread_context",
                },
                "messages_summarized": len(summarized_rows),
                "messages_preserved": len(keep_rows),
                "summary_estimated_tokens": estimate_tokens(summary),
                "preservation": preservation_report,
                "rehydration": rehydration,
            },
            created_at=now,
        )
        self.db.add(boundary)
        self.db.flush()
        return boundary

    def _post_compact_tokens(
        self,
        summary: str,
        keep_rows: list[ChatMessage],
        *,
        current_message: str = "",
        rehydration: dict[str, Any] | None = None,
    ) -> int:
        # 正式摘要保留的消息仍会带回其工具结果，估算时必须一起计算，不能只看最终回答正文。
        projected = (
            MessageStore(self.db).context_projection(keep_rows[0].repo_id, keep_rows[0].conversation_id, keep_rows)
            if keep_rows else []
        )
        return (
            estimate_tokens(summary)
            + sum(estimate_tokens(item["content"]) for item in projected)
            + estimate_tokens(current_message)
            + estimate_tokens(dumps_compact(rehydration or {}))
        )

    def _rehydration_state(self, rows: list[ChatMessage]) -> dict[str, Any]:
        working_files: list[str] = []  # 摘要后继续工作时可能需要重读的近期文件路径
        active_skills: list[str] = []  # 运行记录中实际激活过的 Skill 身份线索
        recent_tools: list[str] = []  # 近期调用的工具名，辅助恢复工作现场，不保存完整大结果

        def remember(target: list[str], value: Any, limit: int) -> None:
            clean = str(value or "").strip()
            # 空项和已存在项不重复记录，恢复提示的空间要留给不同线索。
            if not clean or clean in target:
                return
            target.append(clean)
            if len(target) > limit:
                # 超限时移除列表最前面的旧项，保留后面较近期加入的线索。
                del target[0 : len(target) - limit]

        # 逐条检查原消息中的两种轨迹来源：直接 tool_calls 和元数据 agent_events，兼容不同存储形态。
        for row in rows:
            calls = [item for item in (row.tool_calls or []) if isinstance(item, dict)]
            events = [item for item in (row.meta or {}).get("agent_events") or [] if isinstance(item, dict)]
            # 从真正工具调用参数提取文件路径，不能只在助手自然语言里猜测正在编辑哪个文件。
            for call in calls:
                remember(recent_tools, call.get("tool_name"), 8)
                arguments = call.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
            # 事件轨迹还可能提供 Skill 激活信息，补充直接调用列表不含的恢复线索。
            for event in events:
                event_type = str(event.get("type") or "")
                if event_type in {"tool_use", "tool_result"}:
                    remember(recent_tools, event.get("tool_name"), 8)
                arguments = event.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
                if event_type == "skill_activated":
                    remember(active_skills, event.get("skill_name") or event.get("skill_title"), 6)

        return {
            "working_files": working_files,
            "active_skills": active_skills,
            "recent_tools": recent_tools,
            "recovery": {
                "exact_transcript_tool": "devflow_get_thread_context",
                "sealed_session_tool": "devflow_read_session_events",
            },
        }

    def _raw_segment_tokens(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        store = MessageStore(self.db)
        rows = store.active_for_context(repo_id, conversation_id, limit=1000)
        messages = messages_after_compact_boundary(store.context_projection(repo_id, conversation_id, rows), include_boundary=False)
        total = 0  # 当前活动段的累计估算用量；开始前未计入任何消息，因此初始化为 0
        # 每条消息分别累计角色、正文及工具参数，工具参数也是模型输入，不能只计算可见文字。
        for message in messages:
            total += estimate_tokens(str(message.get("role") or ""))
            total += estimate_tokens(str(message.get("content") or ""))
            # 只有存在工具调用时才序列化估算其成本；没有调用不添加空结构的额外用量。
            if message.get("tool_calls"):
                total += estimate_tokens(dumps_compact(message.get("tool_calls")))
        return total

    def _row_to_message(self, row: ChatMessage) -> dict[str, Any]:
        return {
            "id": str(row.id),
            "role": row.role,
            "content": row.content,
            "route": row.route,
            "tool_calls": row.tool_calls or [],
            "meta": row.meta or {},
        }

    def _consecutive_failures(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        boundary = MessageStore(self.db).last_compact_boundary(repo_id, conversation_id)
        query = self.db.query(ChatMessage).filter(
            ChatMessage.repo_id == repo_id,
            ChatMessage.conversation_id == conversation_id,
            ChatMessage.role == "system",
            ChatMessage.route == "context_compaction",
        )
        if boundary and boundary.created_at:
            query = query.filter(ChatMessage.created_at > boundary.created_at)
        count = 0  # 从最近记录往回数连续摘要失败，尚未扫描时数量为 0
        # 按新到旧扫描，最多看熔断上限条；超过上限的精确数量已不影响本轮决策。
        for row in query.order_by(ChatMessage.created_at.desc()).limit(settings.context_max_compaction_failures).all():
            if (row.meta or {}).get("subtype") == "compact_failure":
                count += 1
            else:
                # 遇到非失败记录就终止连续计数，不能跨过成功记录继续累计更早的失败。
                break
        return count

    def _record_failure(self, repo_id: uuid.UUID, conversation_id: uuid.UUID, current_message: str, stage: str, exc: Exception) -> ChatMessage:
        failure = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role="system",
            content=clip_to_token_budget(f"Context compaction failed during {stage}: {exc}", 240),
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": "compact_failure",
                "stage": stage,
                "current_message": clip_to_token_budget(current_message, 120),
            },
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(failure)
        return failure


class SessionSealer:
    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    # 一轮对话完成后，把用户/助手原文、工具事件、用量和诊断封装成 ChatSession。
    # 封存现场用于后续回看/精确回忆，另外再抽取结构化记忆更新，二者不是同一个产物。
    # 记忆抽取失败时仍要保留原始 transcript 并反映警告，不能因为摘要失败就丢掉本轮现场。
    # 更新当前会话记忆时保留范围；长期 MemoryCandidate 的人工批准流程仍在 memory_hub。
    # 调用方负责最终 commit，确保消息、AgentRun、会话计数和封存结果按上层工作单元提交。
    async def seal(
        self,
        *,
        repo: Repository | None,
        conversation: Conversation | None,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        route: str,
        tool_calls: list[dict[str, Any]],
        agent_events: list[dict[str, Any]] | None = None,
        model_usage: list[dict[str, Any]] | None = None,
        context_diagnostics: dict[str, Any] | None = None,
    ) -> ChatSessionModel:
        agent_events = agent_events or []
        model_usage = model_usage or []
        context_diagnostics = context_diagnostics or {}
        fallback_update = fallback_memory_update(
            user_message.content,
            assistant_message.content,
            route,
            tool_calls,
        )
        fallback_digest = str(fallback_update.get("summary") or "")
        session = ChatSession(
            repo_id=repo.id if repo else None,
            conversation_id=conversation.id if conversation else None,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route=route,
            transcript_json={
                "messages": [
                    {"role": "user", "content": user_message.content, "created_at": self._iso(user_message.created_at)},
                    {"role": "assistant", "content": assistant_message.content, "created_at": self._iso(assistant_message.created_at)},
                ],
                "tool_calls": tool_calls,
                "agent_events": agent_events,
                "tool_use_results": [
                    event for event in agent_events if event.get("type") in {"tool_use", "tool_result"}
                ],
                "model_usage": model_usage,
                "context_diagnostics": context_diagnostics,
                "memory_update": fallback_update,
                "sealing": {"phase": "raw_persisted", "errors": []},
            },
            digest=fallback_digest,
            status="sealing",
            sealed_at=None,
        )
        self.db.add(session)
        self.db.flush()

        errors: list[str] = []
        memory_update = fallback_update
        digest = fallback_digest
        try:
            memory_update = await self._build_memory_update(
                user_message.content,
                assistant_message.content,
                route,
                tool_calls,
                fallback_digest,
                repo_id=repo.id if repo else None,
                conversation_id=conversation.id if conversation else None,
            )
            digest = str(memory_update.pop("_digest", "") or memory_update.get("summary") or fallback_digest)
        except Exception as exc:  # model/network failures must not invalidate the completed chat turn
            errors.append(f"memory_extraction_failed: {exc}")
            memory_update = fallback_update
            digest = fallback_digest

        try:
            with self.db.begin_nested():
                self._update_thread_memory(
                    repo.id if repo else None,
                    conversation.id if conversation else None,
                    digest,
                    memory_update,
                )
                capture_memory_candidates(
                    self.db,
                    repo_id=repo.id if repo else None,
                    conversation_id=conversation.id if conversation else None,
                    session_id=session.id,
                    memory_update=memory_update,
                )
                self.db.flush()
        except Exception as exc:  # keep raw ChatSession even when enrichment fails
            errors.append(f"memory_persistence_failed: {exc}")

        transcript = dict(session.transcript_json or {})
        transcript["memory_update"] = {
            key: value for key, value in memory_update.items() if not key.startswith("_")
        }
        transcript["sealing"] = {
            "phase": "completed",
            "errors": errors,
            "evidence_refresh": "deferred_to_next_request",
        }
        session.transcript_json = transcript
        session.digest = digest
        session.status = "sealed_with_warnings" if errors else "sealed"
        session.sealed_at = datetime.now(timezone.utc)
        self.db.flush()
        return session

    # 尝试把一轮原文与工具事实抽成 facts/tasks/preferences/repo_context 等稳定字段。
    # 模型结果要先规范化结构，再与原记忆合并；不可用时通过规则从现有内容提取回退信息。
    # 抽取结果是压缩后的工作记忆，不等于永久正确的项目事实，仍需保留来源与原文可查性。
    async def _build_memory_update(
        self,
        user_message: str,
        assistant_message: str,
        route: str,
        tool_calls: list[dict[str, Any]],
        digest: str,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        fallback = fallback_memory_update(user_message, assistant_message, route, tool_calls)
        if repo_id and conversation_id:
            current_memory: ConversationMemory | ThreadMemory | None = (
                self.db.query(ConversationMemory)
                .filter(ConversationMemory.conversation_id == conversation_id)
                .first()
            )
        else:
            current_memory = (
                self.db.query(ThreadMemory)
                .filter(_repo_filter(ThreadMemory, repo_id))
                .first()
            )
        existing_memory = self._memory_state(current_memory) if current_memory else {}
        prompt = (
            "请增量更新软件工程 Agent 的数据库会话记忆。"
            "输入包含已有记忆和本轮原始记录；返回更新后的完整快照，而不是只返回本轮增量。"
            "只返回 JSON，包含 digest、summary、facts、decisions、open_questions、tasks、"
            "user_preferences、repo_context、citations。summary 必须描述当前工作状态和下一步；"
            "保留重要文件、函数、命令、错误修复和用户纠正，不要编造。"
        )
        parsed = await self.llm.chat_json(
            prompt,
            {
                "route": route,
                "digest": digest,
                "existing_memory": existing_memory,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_calls": tool_calls,
            },
        )
        update = structured_memory_from_model(parsed) if parsed else fallback
        if not update.get("summary"):
            update["summary"] = fallback["summary"]
        if parsed:
            update["_update_mode"] = "snapshot"
            update["_digest"] = str(parsed.get("digest") or parsed.get("summary") or digest)
        return update

    def _memory_state(self, memory: ConversationMemory | ThreadMemory) -> dict[str, Any]:
        return {
            "summary": memory.summary or "",
            "facts": getattr(memory, "facts", None) or [],
            "decisions": memory.decisions or [],
            "open_questions": memory.open_questions or [],
            "tasks": getattr(memory, "tasks", None) or [],
            "user_preferences": getattr(memory, "user_preferences", None) or [],
            "repo_context": getattr(memory, "repo_context", None) or [],
            "citations": getattr(memory, "citations", None) or [],
        }

    def _new_memory(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ConversationMemory | ThreadMemory:
        if repo_id and conversation_id:
            memory = ConversationMemory(
                repo_id=repo_id,
                conversation_id=conversation_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        else:
            memory = ThreadMemory(
                repo_id=repo_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        self.db.add(memory)
        self.db.flush()
        return memory

    def _update_thread_memory(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        digest: str,
        memory_update: dict[str, Any],
    ) -> ConversationMemory | ThreadMemory | None:
        if repo_id and conversation_id:
            memory: ConversationMemory | ThreadMemory | None = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
        else:
            memory = self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        if not memory:
            memory = self._new_memory(repo_id, conversation_id)
        if memory_update.get("_update_mode") == "snapshot":
            merged = {
                **structured_memory_from_model(memory_update),
                "summary": clip_to_token_budget(str(memory_update.get("summary") or digest), 1100),
            }
        else:
            merged = merge_memory(self._memory_state(memory), memory_update, digest, max_tokens=1100)
        memory.summary = merged["summary"]
        memory.facts = merged["facts"]
        memory.decisions = merged["decisions"]
        memory.open_questions = merged["open_questions"]
        memory.tasks = merged["tasks"]
        memory.user_preferences = merged["user_preferences"]
        memory.repo_context = merged["repo_context"]
        memory.citations = merged["citations"]
        memory.sessions_incorporated = (memory.sessions_incorporated or 0) + 1
        return memory

    def _iso(self, value: datetime | None) -> str | None:
        return value.isoformat() if value else None


class MemoryTools:
    def __init__(self, db: Session, repo: Repository | None, conversation: Conversation | None = None) -> None:
        self.db = db
        self.repo = repo
        self.repo_id = repo.id if repo else None
        self.conversation = conversation
        self.conversation_id = conversation.id if conversation else None

    # 面向 Agent 暴露统一证据搜索，围绕 query 返回片段和可追溯来源。
    # 仓库、会话范围来自服务端工具上下文；limit 限制结果量，避免把整库当作一次观察返回。
    # 召回过程同时记录为 RecallEvent，使学员能检查“Agent 究竟查过什么”。
    async def search_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("query") or input_data.get("message") or "")
        results = EvidenceStore(self.db).search(self.repo_id, self.conversation_id, query, limit=int(input_data.get("limit") or 5))
        answer = "No evidence matched this query." if not results else "\n".join(
            f"- [{item['source_type']}] {item['title']}: {item['snippet']}" for item in results
        )
        citations = [{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_search_evidence",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if self.conversation_id else "project",
            mode="hybrid",
            metadata={"limit": int(input_data.get("limit") or 5)},
        )
        return {
            "route": "memory_search",
            "tool_calls": [{"tool_name": "search_evidence", "arguments": {"query": query}, "status": "success"}],
            "answer": answer,
            "citations": citations,
            "results": results,
        }

    # 精确回忆工具：可读取当前会话上下文，也可按 message_id 精确打开某条持久消息。
    # offset/max_chars 是字符窗口，适合长消息分段读取；它们不是 token 数或字节数。
    # 读取来源跨越摘要边界，不能只返回模型当前 recent 投影，否则越早的内容永远找不回。
    # 结果保留标识和范围，调用过程写召回审计，方便确认回答是否真的查过原文。
    async def get_thread_context(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        keyword = str(input_data.get("keyword") or input_data.get("query") or "").strip().lower()
        limit = min(max(int(input_data.get("limit") or 30), 1), 100)
        message_id = self._uuid_or_none(input_data.get("message_id"))
        before_message_id = self._uuid_or_none(input_data.get("before_message_id"))
        after_message_id = self._uuid_or_none(input_data.get("after_message_id"))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 6000), 200), 16000)

        store = MessageStore(self.db)
        if message_id:
            message = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, self.repo_id),
                    _conversation_filter(ChatMessage, self.conversation_id),
                    ChatMessage.id == message_id,
                    ChatMessage.role.in_(["user", "assistant", "tool", "tool_result"]),
                )
                .one_or_none()
            )
            messages = [message] if message and (message.meta or {}).get("subtype") != "preserved_after_compact" else []
        else:
            messages = store.timeline(
                self.repo_id,
                self.conversation_id,
                limit=None,
                before_message_id=before_message_id,
                after_message_id=after_message_id,
            )
            if keyword:
                terms = _tokens(keyword)
                messages = [msg for msg in messages if any(term in msg.content.lower() for term in terms)]
            messages = messages[-limit:]

        lines: list[str] = []
        results: list[dict[str, Any]] = []
        # 整次工具读取共享的剩余字符额度；每返回一段都要扣除，不是每条消息各给一整份。
        remaining_chars = max_chars
        # 逐条读取持久原文并附带偏移信息，供模型在压缩历史之后继续精确取回；达到总额度即停止。
        for message in messages:
            content = message.content or ""
            # 指定单条消息时 offset 才表示该消息的正文偏移；多消息窗口从各自开头读取。
            chunk_offset = offset if message_id else 0
            chunk = content[chunk_offset : chunk_offset + remaining_chars]
            if not chunk and content:
                continue
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(content)
            created_at = message.created_at.isoformat() if message.created_at else ""
            lines.append(
                f"[message_id={message.id} role={message.role} created_at={created_at} chars={chunk_offset}:{chunk_end}/{len(content)}]"
            )
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[内容未读完；继续调用 message_id={message.id}, offset={chunk_end}]"
                )
            results.append(
                {
                    "title": f"{message.role} message",
                    "source_type": "chat_message",
                    "source_id": str(message.id),
                    "snippet": chunk,
                    "metadata": {
                        "role": message.role,
                        "created_at": created_at,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(content),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break

        citations = [
            {"type": "chat_message", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_get_thread_context",
            query=keyword or str(message_id or ""),
            results=results,
            citations=citations,
            scope="conversation",
            mode="exact_transcript",
            metadata={
                "limit": limit,
                "message_id": str(message_id) if message_id else None,
                "before_message_id": str(before_message_id) if before_message_id else None,
                "after_message_id": str(after_message_id) if after_message_id else None,
                "offset": offset,
                "max_chars": max_chars,
            },
        )
        return {
            "route": "thread_context",
            "tool_calls": [{"tool_name": "get_thread_context", "arguments": {"keyword": keyword, "message_id": str(message_id or "")}, "status": "success"}],
            "answer": "\n".join(lines) if lines else "当前会话还没有匹配的消息。",
            "citations": citations,
            "results": results,
        }

    # 按 session_id 读取封存事件，恢复当时的工具调用、公开进展和可选原始 transcript。
    # 先限定仓库/会话与目标记录，再按请求决定是否带长正文，避免一次返回全部封存内容。
    # 会话摘要只帮助导航，用户要精确原文或执行事件时应读取这里的持久字段。
    # 调用也记录召回事件，使“回忆来自哪一次会话”可被核查。
    async def read_session_events(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        session_id = input_data.get("session_id")
        include_transcript = bool(input_data.get("include_transcript", True))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 8000), 400), 20000)
        query = self.db.query(ChatSession).filter(
            _repo_filter(ChatSession, self.repo_id),
            _conversation_filter(ChatSession, self.conversation_id),
        ).order_by(ChatSession.created_at.desc())
        if session_id:
            parsed_session_id = self._uuid_or_none(session_id)
            if parsed_session_id is None:
                sessions = []
            else:
                sessions = query.filter(ChatSession.id == parsed_session_id).limit(1).all()
        else:
            sessions = query.limit(min(max(int(input_data.get("limit") or 5), 1), 20)).all()
        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars
        # 每个封存会话独立附上身份与摘要；需要时追加 transcript，再共同受本次 max_chars 限制。
        for session in sessions:
            header = f"Session {session.id} route={session.route} sealed_at={session.sealed_at}"
            payload = session.digest
            if include_transcript:
                payload += "\ntranscript=" + dumps_compact(session.transcript_json or {})
            chunk_offset = offset if session_id else 0
            chunk = payload[chunk_offset : chunk_offset + remaining_chars]
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(payload)
            lines.append(header)
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[会话记录未读完；继续调用 session_id={session.id}, offset={chunk_end}。]"
                )
            results.append(
                {
                    "title": f"Session {session.route}",
                    "source_type": "session",
                    "source_id": str(session.id),
                    "snippet": chunk,
                    "metadata": {
                        "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
                        "include_transcript": include_transcript,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(payload),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break
        citations = [
            {"type": "session", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_read_session_events",
            query=str(session_id or "latest sessions"),
            results=results,
            citations=citations,
            scope="conversation",
            mode="sealed_transcript",
            metadata={"include_transcript": include_transcript, "offset": offset, "max_chars": max_chars},
        )
        return {
            "route": "session_events",
            "tool_calls": [{"tool_name": "read_session_events", "arguments": {"session_id": str(session_id or "")}, "status": "success"}],
            "answer": "\n\n".join(lines) if lines else "No sealed sessions found yet.",
            "citations": citations,
            "results": results,
        }

    @staticmethod
    def _uuid_or_none(value: Any) -> uuid.UUID | None:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None
