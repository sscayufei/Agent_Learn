# 关系库文本与 Milvus 向量的适配边界：Document/Payload 提供来源和元数据，向量库负责相似度索引。
# 按仓库、来源与维度定位集合/过滤范围；稳定标识帮助更新时替换同一来源，而不是无限重复插入。
# strict 模式与非 strict 清理的失败语义不同；状态检查必须如实区分禁用、不可用和正常。
import json
import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document
from app.services.rag.embeddings import embed_text, embed_texts, embedding_contract, embedding_provider_name
from app.services.rag.source_policy import policy_metadata, should_vectorize

logger = logging.getLogger(__name__)


class MilvusUnavailableError(RuntimeError):
    pass

try:  # pymilvus 在测试和本地演示模式中是可选依赖。
    from pymilvus import DataType, MilvusClient
except Exception:  # pragma: no cover - 依赖缺失时会走到这里。
    DataType = None  # type: ignore[assignment]
    MilvusClient = None  # type: ignore[assignment]


@dataclass
class DocumentPayload:
    source_type: str  # 来源类型，决定走 RAG 还是直接查询策略
    title: str  # 当前切片的可读标题，常参与向量编码
    content: str  # 待保存的片段正文
    metadata: dict[str, Any] = field(default_factory=dict)  # 章节、行号、校验和等回溯线索
    source_id: uuid.UUID | str | None = None  # 原始业务来源 ID，多片可共用，不等于文档切片 ID


def _as_uuid(value: uuid.UUID | str | None) -> uuid.UUID:
    if value is None:
        return uuid.uuid4()
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def stable_source_id(repo_id: uuid.UUID | str, source_type: str, key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"devflow:{repo_id}:{source_type}:{key}")


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _clip_utf8(value: str, max_bytes: int) -> str:
    data = value.encode("utf-8")
    if len(data) <= max_bytes:
        return value
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _json_dumps(value: dict[str, Any], max_chars: int = 4096) -> str:
    text = json.dumps(value or {}, ensure_ascii=False, default=str)
    return _clip_utf8(text, max_chars)


class MilvusVectorStore:
    def __init__(self) -> None:
        if MilvusClient is None or DataType is None:
            raise RuntimeError("pymilvus is not installed")
        self.collection_name = settings.milvus_collection
        self.dimensions = settings.embedding_dimensions
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        self._ensure_collection()

    # 集合存在时读取 schema 的真实 embedding 维度，并与当前全局维度比较。
    # 不一致就给出换集合/重建提示，不能把新长度向量塞进旧结构或静默改写 schema。
    # 新集合使用显式文档主键、仓库/来源字段、文本和元数据字段，再建 FLOAT_VECTOR 与 COSINE 索引。
    # enable_dynamic_field 允许额外字段，但不取消 embedding 的维度契约。
    # 仓库隔离通过过滤表达式完成，不是这里为每个仓库各自创建不同集合。
    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            description = self.client.describe_collection(self.collection_name)
            actual_dimensions = _collection_dimensions(description)
            # 集合结构属于已存索引契约，配置变化不会自动把旧集合变成新维度，必须显式处理不兼容。
            if actual_dimensions != self.dimensions:
                raise RuntimeError(
                    f"Milvus collection '{self.collection_name}' has dimension {actual_dimensions}; "
                    f"the configured embedding contract requires {self.dimensions}. "
                    "Use a new collection name and rebuild the index."
                )
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="repo_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=80)
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimensions)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    # 文档与向量必须等长，逐条检查来源策略和向量维度，再组织 Milvus 行。
    # pk 使用关系库 Document UUID，同文档再次 upsert 可更新对应索引项，而不是无关联地追加一份。
    # title/content/metadata 按字段容量处理；UTF-8 裁剪按字节边界保护，避免中文被截断成非法字节。
    # repo_id/source_type/source_id 同时进入行，后续检索、删除和来源追溯才能限定范围。
    # 本方法只操作 Milvus，不提交 SQLAlchemy 事务，外部存储成功也不意味着关系库已 commit。
    def upsert_documents(self, documents: list[Document], vectors: list[list[float]]) -> None:
        if len(documents) != len(vectors):
            raise ValueError("document/vector count mismatch")
        rows = []
        # 一条文档和同位置向量组成一行，前面的数量校验防止 zip 静默截断；随后还需逐条检查来源策略与维度。
        for doc, vector in zip(documents, vectors):
            if not should_vectorize(doc.source_type):
                continue
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"document {doc.id} has {len(vector)} embedding dimensions; expected {self.dimensions}"
                )
            rows.append(
                {
                    "pk": str(doc.id),
                    "repo_id": str(doc.repo_id),
                    "document_id": str(doc.id),
                    "source_type": doc.source_type,
                    "source_id": str(doc.source_id),
                    "title": _clip_utf8(doc.title, 500),
                    "content": _clip_utf8(doc.content or "", 8192),
                    "metadata_json": _json_dumps(doc.meta or {}),
                    "embedding": vector,
                }
            )
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)

    # 将 document IDs、仓库、来源类型和来源 ID 组装成 AND 连接的限定条件。
    # 字符串值先转义，避免引号/反斜杠改变过滤表达式结构。
    # 只有存在过滤条件才发删除请求，空条件不应变成误删整个集合。
    # 删除范围越广影响越大；上层必须传入已确认范围，不能让模型直接提交任意全库清理。
    def delete_documents(
        self,
        *,
        repo_id: uuid.UUID | str | None = None,
        source_types: list[str] | None = None,
        source_id: uuid.UUID | str | None = None,
        document_ids: list[uuid.UUID | str] | None = None,
    ) -> None:
        expressions: list[str] = []
        if document_ids:
            ids = ", ".join(f'"{_escape_filter_value(str(item))}"' for item in document_ids)
            expressions.append(f"document_id in [{ids}]")
        if repo_id is not None:
            expressions.append(f'repo_id == "{_escape_filter_value(str(repo_id))}"')
        if source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        if source_id is not None:
            expressions.append(f'source_id == "{_escape_filter_value(str(source_id))}"')
        # 没有条件就不发 delete，保留一道避免空过滤扩大为全量删除的保护。
        if expressions:
            self.client.delete(collection_name=self.collection_name, filter=" and ".join(expressions))

    # 查询时把 repo_id 与可选来源过滤直接交给 Milvus，不先全库检索后再靠前端隐藏。
    # data 是一个查询向量的批次，因此结果通常取 results[0]；limit 限制当前候选数量。
    # 取回 document/source IDs、文本和 metadata_json，再转换成项目统一候选结构。
    # 这里 COSINE 索引返回的 distance 字段被用于相似度排序，不能看到名字 distance 就假定越小越好。
    # _raw_content 留给重排/上下文处理，snippet 只是短展示；保存内容受集合字段容量限制。
    def search(
        self,
        *,
        repo_id: uuid.UUID | str,
        query_embedding: list[float],
        source_type: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        expressions = [f'repo_id == "{_escape_filter_value(str(repo_id))}"']
        if source_type:
            expressions.append(f'source_type == "{_escape_filter_value(source_type)}"')
        elif source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            filter=" and ".join(expressions),
            limit=limit,
            output_fields=["document_id", "source_type", "source_id", "title", "content", "metadata_json"],
        )
        hits = results[0] if results else []
        output: list[dict[str, Any]] = []
        # 逐个把 Milvus 命中转换成业务结构，并解析保存的元数据；检索库行不是 ORM 文档对象。
        for hit in hits:
            entity = hit.get("entity") or {}
            metadata_text = entity.get("metadata_json") or "{}"
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                metadata = {}
            output.append(
                {
                    "id": str(entity.get("document_id") or hit.get("id")),
                    "title": str(entity.get("title") or ""),
                    "source_type": str(entity.get("source_type") or ""),
                    "source_id": str(entity.get("source_id") or ""),
                    "score": round(float(hit.get("distance") or 0.0), 4),
                    "metadata": metadata,
                    "snippet": str(entity.get("content") or "")[:260],
                    "_raw_content": str(entity.get("content") or ""),
                }
            )
        return output


def _collection_dimensions(description: dict[str, Any]) -> int:
    # 从真实 collection schema 找 embedding 字段的维度，不能仅相信本地配置而忽略旧集合的实际形状。
    for schema_field in description.get("fields") or []:
        if schema_field.get("name") != "embedding":
            continue
        params = schema_field.get("params") or schema_field.get("type_params") or {}
        value = params.get("dim") or schema_field.get("dim")
        if value is not None:
            return int(value)
    raise RuntimeError("Milvus collection does not expose an embedding dimension")


# 延迟创建并缓存一个进程内 MilvusVectorStore，避免每次查询都重新连接和检查集合。
# milvus_enabled=False 直接报明确不可用；初始化失败包装为 MilvusUnavailableError 供路由/混合检索处理。
# 配置切换后旧实例仍可能在缓存，测试或明确重配时可使用 reset_milvus_store_cache。
# 缓存只复用连接对象，不缓存查询答案，也不改变每次检索的仓库过滤。
@lru_cache(maxsize=1)
def _milvus_store() -> MilvusVectorStore:
    if not settings.milvus_enabled:
        raise MilvusUnavailableError("Milvus is disabled; vector and hybrid RAG are unavailable")
    try:
        return MilvusVectorStore()
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError(f"Milvus is unavailable: {exc}") from exc


def reset_milvus_store_cache() -> None:
    _milvus_store.cache_clear()


def milvus_runtime_status() -> dict[str, Any]:
    try:
        store = _milvus_store()
    except MilvusUnavailableError as exc:
        return {
            "backend": "milvus",
            "status": "degraded",
            "available": False,
            "error": str(exc),
            "collection": settings.milvus_collection,
            "embedding": embedding_contract(),
        }
    return {
        "backend": "milvus",
        "status": "ready",
        "available": True,
        "error": None,
        "collection": store.collection_name,
        "embedding": embedding_contract(),
    }


# 公共适配入口把可向量化文档与向量送到 Milvus，并将服务错误转换成统一异常。
# 关系库调用方据此决定整项任务是否失败，而不是忽略向量写入错误后标记完成。
# “支持列表输入”不意味着可以把不同维度/不同模型空间的记录混成同一批。
def upsert_milvus_documents(documents: list[Document], vectors: list[list[float]]) -> bool:
    if len(documents) != len(vectors):
        raise ValueError("document/vector count mismatch")
    # 批次写入前每条向量都检查长度，一条异常就拒绝当前写入，不靠补零/截断悄悄凑成合法形状。
    for vector in vectors:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(
                f"embedding has {len(vector)} dimensions; expected {settings.embedding_dimensions}. "
                "Refusing to truncate or pad vectors."
            )
    pairs = [(doc, vector) for doc, vector in zip(documents, vectors) if should_vectorize(doc.source_type)]
    eligible_documents = [item[0] for item in pairs]
    eligible_vectors = [item[1] for item in pairs]
    if not eligible_documents:
        return True
    store = _milvus_store()
    try:
        store.upsert_documents(eligible_documents, eligible_vectors)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to write documents to Milvus") from exc
    return True


def delete_milvus_documents(
    *,
    repo_id: uuid.UUID | str | None = None,
    source_types: list[str] | None = None,
    source_id: uuid.UUID | str | None = None,
    document_ids: list[uuid.UUID | str] | None = None,
    strict: bool = True,
) -> bool:
    try:
        store = _milvus_store()
    except MilvusUnavailableError:
        if strict:
            raise
        return False
    try:
        store.delete_documents(repo_id=repo_id, source_types=source_types, source_id=source_id, document_ids=document_ids)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        if not strict:
            logger.warning("从 Milvus 删除文档失败：%s", exc)
            return False
        raise MilvusUnavailableError("failed to delete documents from Milvus") from exc
    return True


def search_milvus_documents(
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 20,
    query_embedding: list[float] | None = None,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    store = _milvus_store()
    try:
        vector = query_embedding or embed_text(query, settings.embedding_dimensions)
        return store.search(
            repo_id=repo_id,
            query_embedding=vector,
            source_type=source_type,
            source_types=source_types,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to search documents in Milvus") from exc


def add_document(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    source_id: str | uuid.UUID | None,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    commit: bool = True,
) -> Document:
    if source_type == "code_file":
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")
    vectorized = should_vectorize(source_type)
    vectors = [embed_text(f"{title}\n{content}", settings.embedding_dimensions)] if vectorized else []
    embedding_metadata = (
        {
            "embedding_provider": embedding_provider_name(),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        }
        if vectorized
        else {}
    )
    doc = Document(
        repo_id=_as_uuid(repo_id),
        source_type=source_type,
        source_id=_as_uuid(source_id),
        title=title[:500],
        content=content,
        meta={
            **(metadata or {}),
            **policy_metadata(source_type),
            **embedding_metadata,
        },
    )
    db.add(doc)
    db.flush()
    if vectorized and settings.milvus_enabled:
        upsert_milvus_documents([doc], vectors)
    if commit:
        db.commit()
        db.refresh(doc)
    return doc


# 批量写入关系库文档，可接收预生成向量，也可以按来源策略自行编码。
# 先清掉空正文并拒绝 code_file；当前源码必须走工作区实时读取，不能绕过策略写进 RAG Document。
# vector_indexes 记录需要向量化条目在完整列表中的位置，混合 direct/rag 来源也能保持位置对应。
# 提供 vectors 时检查完整批次长度；需要同步却缺向量时提前报错，防止无声漏索引。
# 创建 Document 并 flush 后得到 UUID，再将向量化子集与对应向量同步 Milvus。
# commit=True 才提交/刷新关系库对象，False 允许上层把多步组合为同一关系库事务。
# 注意跨数据库边界：SQL rollback 不会自动撤销已发出的 Milvus 请求。
def add_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    # 后续向量数量要与清理后的有效正文对应，调用方预生成向量时也需要保持同一批次约定。
    clean_payloads = [payload for payload in payloads if payload.content.strip()]
    if any(payload.source_type == "code_file" for payload in clean_payloads):
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")
    if vectors is not None and len(vectors) != len(clean_payloads):
        raise ValueError("embedding count does not match document payload count")

    embeddings: list[list[float] | None] = [None] * len(clean_payloads)
    # 只给允许进入 RAG 的来源生成向量；用原列表索引回填，保证混合来源批次仍一一对齐。
    vector_indexes = [
        index for index, payload in enumerate(clean_payloads) if should_vectorize(payload.source_type)
    ]
    if vectors is not None:
        # 已有向量时只回填允许向量化来源的原索引，避免删除 direct 来源后让列表位置整体错位。
        for index in vector_indexes:
            embeddings[index] = vectors[index]
    elif vector_indexes and generate_embeddings:
        generated = embed_texts(
            [f"{clean_payloads[index].title}\n{clean_payloads[index].content}" for index in vector_indexes],
            settings.embedding_dimensions,
        )
        if len(generated) != len(vector_indexes):
            raise ValueError("embedding count does not match vectorized document payload count")
        # 只把生成结果回填到 RAG 来源的原位置，团队等 direct 来源不应被迫生成没有用途的向量。
        for index, embedding in zip(vector_indexes, generated):
            embeddings[index] = embedding
    # 要求同步向量库时，所有可向量化条目必须已有向量；缺失不能静默跳过后报告成功。
    if sync_vector_store and any(embeddings[index] is None for index in vector_indexes):
        raise ValueError("RAG documents require embeddings before Milvus synchronization")

    docs: list[Document] = []
    # 顺序一一配对正文与向量，direct 来源可没有向量但仍可存关系库文档，二者不能混成必须全向量化。
    for payload, embedding in zip(clean_payloads, embeddings):
        vectorized = embedding is not None
        embedding_metadata = (
            {
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                **(embedding_info or {}),
            }
            if vectorized
            else {}
        )
        docs.append(
            Document(
                repo_id=_as_uuid(repo_id),
                source_type=payload.source_type,
                source_id=_as_uuid(payload.source_id),
                title=payload.title[:500],
                content=payload.content,
                meta={
                    **(payload.metadata or {}),
                    **policy_metadata(payload.source_type),
                    **embedding_metadata,
                },
            )
        )
    if not docs:
        return []
    db.add_all(docs)
    db.flush()
    if sync_vector_store and settings.milvus_enabled:
        # 同步子集与生成向量使用同一组索引，避免“第几个可向量化条目”与“原批次第几项”混淆。
        vector_documents = [docs[index] for index in vector_indexes]
        vector_values = [embeddings[index] for index in vector_indexes]
        upsert_milvus_documents(
            vector_documents,
            [vector for vector in vector_values if vector is not None],
        )
    if commit:
        db.commit()
        # 提交后逐个刷新数据库生成字段，返回的 UUID/时间等才与真实持久记录一致；flush 路径由上层统一提交。
        for doc in docs:
            db.refresh(doc)
    return docs


# 对一个来源类型的文档组执行替换：先建立明确删除范围，再添加新 payload。
# 使用稳定来源/仓库约束，避免更新某类资料时顺带清掉其他项目或其他来源。
# 替换涉及旧向量清理和关系库新写入，失败恢复不能只看最后一个 SQL commit 是否成功。
def replace_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    source_types: list[str],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    repo_uuid = _as_uuid(repo_id)
    db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type.in_(source_types)).delete(
        synchronize_session=False
    )
    db.flush()
    if sync_vector_store and settings.milvus_enabled:
        delete_milvus_documents(repo_id=repo_uuid, source_types=source_types)
    docs = add_documents(
        db,
        repo_uuid,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    if commit:
        db.commit()
    return docs


# 按指定元数据范围替换文档组，适合只刷新某个子集合而非整仓库来源。
# 上层需要提供明确、稳定的筛选条件，并核对它与新 payload 对应。
# 保持业务定位与存储细节分开，调用方不用自己散落拼接向量删除逻辑。
def replace_documents_by_metadata(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    metadata_filters: dict[str, Any],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
) -> list[Document]:
    repo_uuid = _as_uuid(repo_id)
    rows = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type == source_type).all()
    document_ids = [
        row.id
        for row in rows
        if all((row.meta or {}).get(key) == value for key, value in metadata_filters.items())
    ]
    if document_ids:
        db.query(Document).filter(Document.id.in_(document_ids)).delete(synchronize_session=False)
        if should_vectorize(source_type) and settings.milvus_enabled:
            delete_milvus_documents(repo_id=repo_uuid, document_ids=document_ids)
    db.flush()
    docs = add_documents(db, repo_uuid, payloads, commit=False)
    if commit:
        db.commit()
    return docs


# 按仓库、source_id 与可选 source_type 清理同一原始来源拆出的多个片段。
# 原文件与 chunk 是一对多语义关系，删除来源不能只删除其中第一片。
# commit 参数允许后台重处理先删旧组、加新组，再统一提交关系库状态。
def delete_document_group(
    db: Session,
    repo_id: str | uuid.UUID,
    source_id: str | uuid.UUID,
    source_type: str | None = None,
    *,
    commit: bool = True,
) -> int:
    repo_uuid = _as_uuid(repo_id)
    source_uuid = _as_uuid(source_id)
    query = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_id == source_uuid)
    if source_type:
        query = query.filter(Document.source_type == source_type)
    document_ids = [row.id for row in query.all()]
    count = query.delete(synchronize_session=False)
    if settings.milvus_enabled and (source_type is None or should_vectorize(source_type)):
        delete_milvus_documents(repo_id=repo_uuid, source_id=source_uuid, document_ids=document_ids)
    if commit:
        db.commit()
    return count
