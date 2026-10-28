# 上传文档的后台处理流水线：parsing -> splitting -> embedding -> completed，异常进入 failed。
# 原文件记录、关系库切片与 Milvus 向量是三个层次；状态提交让前端能观察耗时任务的进度。
# 数据库事务与外部向量库不是同一个事务，排查失败时要分别核对文档状态、切片和向量索引。
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db.models import KnowledgeSourceDocument
from app.db.session import SessionLocal
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.knowledge_base import get_or_create_config
from app.services.rag.vector_store import DocumentPayload, add_documents, delete_document_group

SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown", ".mdx", ".pdf", ".docx", ".json", ".csv", ".log"}
EMBEDDING_BATCH_SIZE = 64


def extract_text(filename: str, data: bytes) -> str:
    return extract_document_text(filename, data)


# 把输入文本按 EMBEDDING_BATCH_SIZE 分批调用 Embedding，限制单次请求体和模型内存压力。
# 每个批次沿用相同 provider/model/dimensions，输出顺序必须严格对应原文本顺序。
# extend 按批次拼回总向量列表，下游再与标题、内容和 chunk 元数据按相同索引配对。
# 任一批次异常会抛给处理任务统一记录失败，不把缺向量的半成品伪装成 completed。
def _embed_in_batches(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []  # 最终向量按原文本顺序累积；开始还没编码任何批次，所以先为空
    # 从第 0 条起每次前进一个批量大小，避免一次请求放入全部文档；最后不足一批也照常处理。
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        # extend 把本批每条向量接到总列表；若用 append 会多套一层批次列表，破坏文本与向量的一一对应。
        vectors.extend(
            embed_knowledge_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
        )
    return vectors


# 用独立短 Session 更新来源状态并立即提交，使前端能在后台任务开始阶段看到进展。
# 目标记录已被删除时直接返回，避免为不存在的原文件重新造一条状态记录。
# finally 始终关闭 Session，后台线程/进程不能依赖已经结束的上传请求 Session。
def _set_status(document_id: uuid.UUID, status: str, *, error: str | None = None) -> None:
    db = SessionLocal()
    try:
        document = db.get(KnowledgeSourceDocument, document_id)
        if document is None:
            return
        document.status = status
        document.error_message = error
        document.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


# 以原文件记录 UUID 为输入，后台任务自己创建 Session，并从 stored_path 重新读取受限文件内容。
# parsing：提取文本并确认非空；splitting：按知识库大小/重叠配置产生带来源定位的 chunk。
# embedding：为每个“标题 + 正文”生成向量，保留与 chunk 同序的标题和元数据。
# 准备 payload：保存原文件 ID、checksum、文件名、章节/行号/父子关系，便于检索后回原文。
# 写入：替换该来源旧文档组并批量添加新切片/向量，最后把原文件状态提交为 completed。
# 失败：回滚关系库当前事务，记录截断后的错误信息；finally 释放本任务数据库连接。
# 状态中间提交方便观测，但数据库与 Milvus 是两个存储系统，失败后仍可能需要重试/重建核对一致性。
def process_knowledge_document(document_id: uuid.UUID) -> None:
    _set_status(document_id, "parsing")
    db = SessionLocal()
    try:
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is None:
            return
        # 读取上限多一个字节用于判断超限：若只读恰好 MAX 字节，就分不清文件刚好到上限还是仍有后续。
        with Path(source.stored_path).open("rb") as handle:
            data = handle.read(MAX_DOCUMENT_BYTES + 1)
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError(f"knowledge file exceeds the {MAX_DOCUMENT_BYTES} byte limit")
        text = extract_text(source.name, data)
        if not text.strip():
            raise ValueError("文档中没有可索引的文本")

        # 先提交阶段再做耗时工作，前端轮询才不会整个过程一直停留在 queued。
        source.status = "splitting"
        source.character_count = len(text)
        db.commit()
        # 使用该仓库保存的有效知识库设置；Embedding 入口仍会校验它与共享向量空间契约一致。
        config = get_or_create_config(db, source.repo_id)
        chunks = chunk_document(
            text,
            source.name,
            max_chars=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        if not chunks:
            raise ValueError("文档切分后没有有效片段")

        # 此时切片已成功，但尚未保证所有向量生成/入库成功，不能提前显示“可用”。
        source.status = "embedding"
        source.chunk_count = len(chunks)
        db.commit()
        titles = []  # 每个切片的展示/编码标题，后面保持与 chunks、contents、vectors 同一索引顺序
        # 切片不能只共用一个文件标题：逐片优先用章节名，再退回页号或片号，便于检索后定位。
        for index, chunk in enumerate(chunks):
            label = chunk.get("section_title")
            if not label and chunk.get("page"):
                label = f"Page {chunk['page']}"
            # 内部索引从 0 起，但展示片号从 1 起，便于用户阅读；不改变实际 chunk 身份。
            label = str(label or f"Chunk {index + 1}")
            child_suffix = f" ({int(chunk.get('child_index') or 0) + 1})" if chunk.get("child_index") else ""
            titles.append(f"{source.name} · {label}{child_suffix}")
        # 下面 titles/contents/chunks 采用同一顺序；任意一组独立重排序都会把向量配到错误正文上。
        # 按相同切片顺序提取正文，不在这里单独排序，否则后面的 zip 会配错标题与内容。
        contents = [str(chunk["content"]) for chunk in chunks]
        vectors = _embed_in_batches(
            [f"{title}\n{content}" for title, content in zip(titles, contents)],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        # 把每片正文、原文件身份、哈希和定位信息组装为存储输入；不是只存向量而丢掉可阅读文本。
        payloads = [
            DocumentPayload(
                source_type="knowledge_file",
                source_id=source.id,
                title=title,
                content=content,
                metadata={
                    **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                    "display_title": source.name,
                    "filename": source.name,
                    "knowledge_document_id": str(source.id),
                    "checksum": source.checksum,
                    "file_size": source.file_size,
                    "content_type": source.content_type,
                    "chunk_index": index,
                    "chunk_chars": len(content),
                    "total_chunks": len(chunks),
                    "chunk_strategy": "structure_aware_v1",
                },
            )
            for index, (title, content, chunk) in enumerate(zip(titles, contents, chunks))
        ]
        # 全部文本和向量准备后才替换该原文件的旧片段；commit=False 让上层控制关系库提交点。
        delete_document_group(db, source.repo_id, source.id, source_type="knowledge_file", commit=False)
        add_documents(
            db,
            source.repo_id,
            payloads,
            commit=False,
            vectors=vectors,
            embedding_info={
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model,
                "embedding_dimensions": config.embedding_dimensions,
            },
        )
        # 只有写入路径正常返回才标记完成；解析成功、切片成功或拿到部分向量都不足以算完成。
        source.status = "completed"
        source.error_message = None
        source.completed_at = datetime.now(timezone.utc)
        source.updated_at = source.completed_at
        db.commit()
    except Exception as exc:
        # 回滚关系库当前事务后，再保存失败状态供前端重试；这不等于自动回滚外部 Milvus 的全部操作。
        db.rollback()
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is not None:
            source.status = "failed"
            # 给学员保留足够诊断信息但限制存储长度；原始输入和真实密钥不应主动拼进错误文本。
            source.error_message = str(exc)[:1000]
            source.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
