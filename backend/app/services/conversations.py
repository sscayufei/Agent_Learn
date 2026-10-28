# 仓库下的独立会话管理：默认会话兼容旧数据，新会话保存各自消息与记忆。
# 删除会话需处理关联记录；默认会话回填只用于兼容，不应把不同会话的历史混在一起。
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    ChatFeedback,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    EvidenceItem,
    MemoryCandidate,
    RecallEvent,
    Repository,
)


def _default_title(db: Session, repo_id: uuid.UUID) -> str:
    existing = db.query(Conversation).filter(Conversation.repo_id == repo_id).count()
    return "默认会话" if existing == 0 else f"新会话 {existing + 1}"


def ensure_default_conversation(db: Session, repo_id: uuid.UUID) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.created_at.asc())
        .first()
    )
    if conversation:
        return conversation
    conversation = Conversation(repo_id=repo_id, title="默认会话", status="active")
    db.add(conversation)
    db.flush()
    return conversation


# 有 conversation_id 时核对它属于当前仓库；没有时走默认会话兼容路径。
# 同一仓库可有多个会话，不能只按 repo_id 读取并混合全部历史。
# 确保会话只是定位/创建对象，最终提交由调用路径控制。
def ensure_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID | None = None) -> Conversation:
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
            .one_or_none()
        )
        if conversation:
            return conversation
    return ensure_default_conversation(db, repo_id)


def list_conversations(db: Session, repo_id: uuid.UUID) -> list[Conversation]:
    ensure_default_conversation(db, repo_id)
    _backfill_repo_conversation(db, repo_id)
    db.flush()
    return (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .all()
    )


def create_conversation(db: Session, repo_id: uuid.UUID, title: str | None = None) -> Conversation:
    conversation = Conversation(repo_id=repo_id, title=(title or "").strip() or _default_title(db, repo_id))
    db.add(conversation)
    db.flush()
    return conversation


# 清理该会话关联的消息、封存、记忆、证据/运行等记录，保留同仓库其他会话。
# 多处使用 conversation_id 的模型都要纳入范围，不能只删除 Conversation 行留下孤儿数据。
# 默认会话兼容/回填与用户主动删除语义要区分，不能删除后悄悄合并到别的会话。
def delete_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
        .one_or_none()
    )
    if conversation is None:
        return ensure_default_conversation(db, repo_id)

    session_ids = [
        item.id
        for item in db.query(ChatSession.id).filter(ChatSession.conversation_id == conversation.id).all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id).filter(AgentWorkflowRun.conversation_id == conversation.id).all()
    ]

    db.query(ChatFeedback).filter(ChatFeedback.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(MemoryCandidate.conversation_id == conversation.id).delete(synchronize_session=False)
    if session_ids:
        db.query(MemoryCandidate).filter(MemoryCandidate.session_id.in_(session_ids)).delete(synchronize_session=False)
    db.query(RecallEvent).filter(RecallEvent.conversation_id == conversation.id).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(AgentWorkflowRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(AgentRun).filter(AgentRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(EvidenceItem.conversation_id == conversation.id).delete(synchronize_session=False)
    conversation.status = "deleted"
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()

    replacement = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .first()
    )
    return replacement or create_conversation(db, repo_id, "默认会话")


def touch_conversation(
    db: Session,
    conversation: Conversation,
    *,
    user_message: str | None = None,
    increment: int = 0,
) -> None:
    if user_message and conversation.message_count == 0 and conversation.title.startswith(("默认会话", "新会话")):
        title = user_message.strip().replace("\n", " ")
        if title:
            conversation.title = title[:40]
    conversation.message_count = (conversation.message_count or 0) + increment
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()


def backfill_all_default_conversations(db: Session) -> None:
    # 旧数据按仓库分别补默认会话，不能让所有仓库消息归到同一个全局会话。
    for repo in db.query(Repository).all():
        ensure_default_conversation(db, repo.id)
        _backfill_repo_conversation(db, repo.id)
    db.commit()


def _backfill_repo_conversation(db: Session, repo_id: uuid.UUID) -> None:
    conversation = ensure_default_conversation(db, repo_id)
    # 消息、封存会话、运行轨迹三类旧记录都需要补会话身份，只更新当前仓库且尚无会话的行。
    for model in [ChatMessage, ChatSession, AgentRun]:
        db.query(model).filter(model.repo_id == repo_id, model.conversation_id.is_(None)).update(
            {model.conversation_id: conversation.id},
            synchronize_session=False,
        )
    conversation.message_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).count()
