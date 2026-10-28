# RAG 入库边界的集中定义：反复复用的非结构化知识进入语义检索，当前源码和实时结构化状态不混入。
# 源码走当前 checkout 的词法/文件工具，团队与实时状态走数据库/API；向量命中只是候选证据。
from collections.abc import Iterable
from typing import Literal


RetrievalMode = Literal["rag", "direct"]

# RAG is reserved for reusable, unstructured project knowledge where semantic
# recall is useful. Structured live facts remain in their source tables/APIs.
RAG_SOURCE_TYPES = frozenset(
    {
        "issue",
        "pull_request",
        "workflow_run",
        "project_doc",
        "knowledge_file",
        "memory_note",
    }
)

DIRECT_SOURCE_TYPES = frozenset(
    {
        "project_overview",
        "project_manifest",
        "team_member",
        "weekly_report",
        "chat_session",
        "conversation_memory",
        "thread_memory",
    }
)


# 把来源类型归类为 RAG、直接查询或其他模式，集中表达本项目的知识边界。
# Issue 描述等反复利用的非结构化资料适合语义召回；实时状态/团队字段用结构化查询更可靠。
# 这个分类供入库与读取共同使用，避免写入层允许而检索层又悄悄使用另一套来源清单。
def retrieval_mode(source_type: str) -> RetrievalMode:
    if source_type in RAG_SOURCE_TYPES:
        return "rag"
    return "direct"


def normalize_source_types(source_types: Iterable[str | None] | None) -> list[str]:
    normalized = [str(source_type).strip() for source_type in (source_types or []) if source_type and str(source_type).strip()]
    return list(dict.fromkeys(normalized))


# 回答“这个来源是否应该生成向量”，不是回答“这个记录是否应该保存在数据库”。
# direct 来源仍可保存结构化记录，但不应为了统一形式就占用向量索引。
# 源码 code_file 的实时读取路径同样不靠历史向量证明当前文件内容。
def should_vectorize(source_type: str) -> bool:
    return retrieval_mode(source_type) == "rag"


def is_searchable_source(source_type: str) -> bool:
    return retrieval_mode(source_type) == "rag"


# 将来源的检索模式等策略写成可附在文档上的元数据，便于状态页和回归检查。
# 元数据是策略可观察投影，不应被用来绕过真正的 should_vectorize/is_searchable_source 判断。
def policy_metadata(source_type: str) -> dict[str, object]:
    mode = retrieval_mode(source_type)
    return {
        "retrieval_mode": mode,
        "rag_enabled": mode == "rag",
    }
