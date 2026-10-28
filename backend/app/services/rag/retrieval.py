# 混合检索主链路：按仓库/来源过滤 -> 向量与 BM25 召回 -> 加权融合 -> 去重/父文档限额 -> 重排 -> 压缩。
# 分数是排序信号，不是“答案正确率”；向量与关键词先分别归一化，融合时再保留每路的诊断轨迹。
# hybrid 下 Milvus 不可用可以保留关键词结果，但需要报告 unavailable；纯 vector 请求则直接失败。
import heapq
import hashlib
import math
import re
import uuid
from collections import Counter
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.context_compression import compress_retrieval_results
from app.services.rag.rerank import rerank_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES, is_searchable_source, normalize_source_types
from app.services.rag.vector_store import MilvusUnavailableError, search_milvus_documents


BM25_K1 = 1.5  # 控制词频饱和，重复关键词不会无限加分
BM25_B = 0.75  # 控制按文档长度归一化的程度
VECTOR_FUSION_WEIGHT = 0.62  # 向量归一化分在混合公式中的权重
KEYWORD_FUSION_WEIGHT = 0.30  # 关键词归一化分在混合公式中的权重
DUAL_RECALL_BONUS = 0.08  # 同片段被两路正分命中时的额外奖励
MAX_CHUNKS_PER_PARENT = 2  # 单个父文档最多占几个候选名额
NEAR_DUPLICATE_JACCARD = 0.9  # 同来源近重复词集合的相似阈值


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _parse_datetime(value: str | None, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        if len(value) == 10:
            parsed_date = datetime.fromisoformat(value).date()
            return datetime.combine(parsed_date, time.max if end_of_day else time.min, tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _metadata_time(doc_meta: dict) -> datetime | None:
    for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
        parsed = _parse_datetime(doc_meta.get(key))
        if parsed:
            return parsed
    return None


# 统一处理标签、状态、路径、时间、分支及其他元数据过滤，让两路召回的范围尽量一致。
# 开始/结束日期支持只给日期，结束日期扩展到当天末尾；缺失必要时间信息的候选不能冒充满足范围。
# 列表字段按成员关系匹配，普通字段按相等匹配；path 会结合文件列表或可用元数据定位。
# 这一步是检索过滤，不是用户权限体系；调用入口还应验证请求可访问的仓库。
def _matches_metadata(doc_meta: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    label = filters.get("label")
    if label and label not in (doc_meta.get("labels") or []):
        return False

    state = filters.get("state")
    if state and doc_meta.get("state") != state:
        return False

    path = (filters.get("path") or filters.get("path_prefix") or filters.get("module") or "").lower()
    if path:
        files = [str(item).lower() for item in doc_meta.get("files") or []]
        if files and not any(path in filename for filename in files):
            return False
        if not files and path not in " ".join(str(value).lower() for value in doc_meta.values()):
            return False

    start_at = _parse_datetime(filters.get("start_date") or filters.get("updated_after"))
    end_at = _parse_datetime(filters.get("end_date") or filters.get("updated_before"), end_of_day=True)
    doc_time = _metadata_time(doc_meta)
    if start_at and (doc_time is None or doc_time < start_at):
        return False
    if end_at and (doc_time is None or doc_time > end_at):
        return False

    for key, value in filters.items():
        if key in {
            "label",
            "path",
            "path_prefix",
            "module",
            "start_date",
            "end_date",
            "updated_after",
            "updated_before",
        } or value in (None, ""):
            continue
        if key == "branch":
            current = doc_meta.get("branch") or doc_meta.get("base_branch") or doc_meta.get("head_branch")
            if current != value:
                return False
            continue
        current = doc_meta.get(key)
        if isinstance(current, list):
            if value not in current:
                return False
        elif current != value:
            return False
    return True


# 将英文/数字/下划线等提取为词项，并给中文连续段补单字与相邻双字。
# 例如“令牌刷新”可产生字与双字组合，减少完全依赖空格分词时中文没有命中的问题。
# 这是轻量词法方法，可能有噪声和同义词不足；语义分支用于补充不同表达间的召回。
def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _clip(text: str, limit: int = 260) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


# 将单路候选非负分数除以该批最大分，映射到相对可比的范围，并写回指定字段。
# BM25 与向量相似度原始尺度不同，先各自归一化再做加权融合，避免一方只因数值大支配排序。
# 这是当前候选集合的相对分数：换了查询或候选池，0.8 不一定表示同样的绝对相关性。
# 因此后续阈值与质量判断应结合检索配置和回答门控，不把归一化分当概率。
def _normalise_scores(items: list[dict[str, Any]], field: str) -> None:
    if not items:
        return
    values = [max(0.0, float(item.get(field) or 0.0)) for item in items]
    # 本批最大值作分母，让不同原始量级便于融合；全为 0 时用 1 防止除零，仍会得到 0。
    max_value = max(values) or 1.0
    # 原值与候选按相同顺序一一对应，原地更新指定分数字段；不是重新安排候选次序。
    for item, value in zip(items, values):
        item[field] = round(value / max_value, 4)


def _document_result(doc: Document, score: float, source: str, score_field: str) -> dict[str, Any]:
    metadata = doc.meta or {}
    result = {
        "id": str(doc.id),
        "title": doc.title,
        "source_type": doc.source_type,
        "source_id": str(doc.source_id),
        "score": round(max(0.0, score), 4),
        "metadata": metadata,
        "snippet": _clip(doc.content),
        "_raw_content": doc.content,
        score_field: round(max(0.0, score), 4),
        "retrieval": {"sources": [source], score_field: round(max(0.0, score), 4)},
    }
    return result


# 调用带 repo/source 约束的 Milvus 检索，再按元数据过滤候选。
# 保留 vector_score_raw，另写归一化 vector_score 与向量排名，用于融合和诊断。
# 数据结构统一包含 id、source_id、metadata、片段与检索轨迹，便于与 BM25 候选合并。
# 查询向量可由上游预先生成并传入，避免同一问题在多个层次重复编码。
def _milvus_vector_candidates(
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    milvus_results = search_milvus_documents(
        repo_id,
        query,
        source_type=source_types[0] if len(source_types) == 1 else None,
        source_types=source_types if len(source_types) > 1 else None,
        limit=fetch_limit,
        query_embedding=query_embedding,
    )
    candidates = []
    # 向量库的每条命中仍需做元数据过滤，再转换为业务候选，不能只凭相似度绕过状态/路径条件。
    for item in milvus_results:
        if not _matches_metadata(item.get("metadata") or {}, metadata_filters):
            continue
        score = max(0.0, float(item.get("score") or 0.0))
        candidates.append(
            {
                **item,
                "id": str(item.get("id") or item.get("source_id") or ""),
                "score": score,
                "vector_score": score,
                "vector_score_raw": score,
                "retrieval": {
                    "sources": ["vector"],
                    "vector_score_raw": score,
                    "vector_backend": "milvus",
                },
            }
        )
    _normalise_scores(candidates, "vector_score")
    # 归一化后同步当前分与向量分，并记录从 1 起的路内排名，后续融合仍能追踪原来的排序。
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["vector_score"]
        item["retrieval"]["vector_score"] = item["vector_score"]
        item["vector_rank"] = rank
        item["retrieval"]["vector_rank"] = rank
    return candidates


# 把一份文档拆成 title/path/symbol/labels/content 五个字段，分别统计查询词频。
# 标题和符号等字段权重更高，稀有错误名/路径的精确命中通常比正文里反复泛提更重要。
# 返回加权词频、实际命中词和文档长度，后续需结合整个过滤后语料的统计计算 BM25。
# 先统计再评分能使用一致的文档频率和平均长度，避免每篇文档自己定义分数尺度。
def _bm25_document_stats(doc: Document, query_terms: list[str]) -> dict[str, Any]:
    metadata = doc.meta or {}
    field_tokens = {
        "title": _tokens(doc.title or ""),
        "path": _tokens(str(metadata.get("path") or "")),
        "symbol": _tokens(str(metadata.get("symbol") or metadata.get("name") or "")),
        "labels": _tokens(" ".join(str(item) for item in metadata.get("labels") or [])),
        "content": _tokens(doc.content or ""),
    }
    field_weights = {"title": 4.0, "path": 3.0, "symbol": 4.5, "labels": 2.5, "content": 1.0}
    counters = {field: Counter(tokens) for field, tokens in field_tokens.items()}
    weighted_tf = {
        term: sum(float(counters[field].get(term, 0)) * field_weights[field] for field in field_tokens)
        for term in query_terms
    }
    matched_terms = [term for term in query_terms if weighted_tf.get(term, 0.0) > 0.0]
    document_length = max(1, sum(len(tokens) for tokens in field_tokens.values()))
    return {
        "doc": doc,
        "weighted_tf": weighted_tf,
        "matched_terms": matched_terms,
        "document_length": document_length,
    }


# BM25 对每个命中查询词累加：稀有词权重 IDF × 饱和词频 / 长度归一化影响。
# document_frequency 越小，词越稀有，IDF 越大；普通泛词不会只因到处出现就获得最高权重。
# 词频分式逐渐饱和，防止把同一个词重复一百次就无限提高排名。
# 长度项使用文档长度相对平均长度，缓解长日志/长文天然拥有更多匹配机会的问题。
# BM25_K1/B 是当前实现参数，不是学习出的答案正确率；结果主要用于排序。
def _bm25_score(
    stats: dict[str, Any],
    *,
    document_count: int,
    document_frequency: Counter[str],
    average_document_length: float,
) -> float:
    score = 0.0  # 当前文档的 BM25 总分；还没有累加查询词贡献时为 0.0，浮点用于后面的连续分数
    document_length = float(stats["document_length"])  # 当前文档的词项长度，用于和整个候选语料的平均长度比较
    # 长文通常更容易出现查询词，因此按“当前长度/平均长度”校正词频优势；平均长度至少取 1 防止除零。
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / max(average_document_length, 1.0))
    # 逐个查询词累加它对本篇文档的贡献；标题/标签等字段影响已经体现在 weighted_tf，不是在这里再重复加权。
    for term, weighted_tf in stats["weighted_tf"].items():
        # 这个词在当前文档没有正贡献，跳过即可，不让无命中的词凭 IDF 产生分数。
        if weighted_tf <= 0:
            continue
        df = float(document_frequency.get(term, 0))  # 含有该词的文档数，不是该词出现的总次数
        idf = math.log1p((document_count - df + 0.5) / (df + 0.5))  # 词越稀有，区分文档的权重越高
        # 用词的稀有度乘以饱和词频项，再加到总分；分母让重复堆词的收益递减，而不是无限线性增加。
        score += idf * (weighted_tf * (BM25_K1 + 1.0)) / (weighted_tf + length_norm)
    # 向后续归一化返回非负排序分，不把原始 BM25 当成 0 到 1 的概率。
    return max(0.0, score)


# 先按当前仓库与允许来源选择关系库语料，并在计算统计前应用状态/元数据范围。
# 遍历语料积累查询词的文档频率与总长度，再基于统一平均长度为每份文档评分。
# 使用固定大小小顶堆保留 fetch_limit 个高分结果，堆顶是当前保留集合中最弱的一项。
# 新候选更强时替换堆顶，避免对全部候选做不受限排序；但当前语料统计本身仍在内存维护。
# 最后归一化关键词分，保留 BM25 原分、命中词与参数，供融合和问题诊断。
def _keyword_candidates(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
) -> list[dict[str, Any]]:
    # 查询词先去重且保持顺序，避免同一个词在问题里多出现几次就重复计入文档评分。
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms:
        return []
    q = db.query(Document).filter(
        Document.repo_id == _as_uuid(repo_id),
        Document.source_type.in_(source_types),
    )
    state = (metadata_filters or {}).get("state")
    if state:
        q = q.filter(Document.meta["state"].as_string() == str(state))

    corpus: list[dict[str, Any]] = []  # 当前过滤范围内的文档统计，供统一 BM25 评分
    document_frequency: Counter[str] = Counter()  # 每个查询词出现于多少份文档，不是总词频
    total_document_length = 0  # 语料总词项量，用于计算平均文档长度
    # 分批取数据库行降低一次性 ORM 拉取压力；后面仍会保存语料统计，所以不是完全常量内存流算法。
    for doc in q.yield_per(500):
        if not _matches_metadata(doc.meta or {}, metadata_filters):
            continue
        stats = _bm25_document_stats(doc, terms)
        corpus.append(stats)
        total_document_length += int(stats["document_length"])
        # 同一篇内某词只记一次文档频率；否则长文的重复词会被误当作命中了多篇文档。
        document_frequency.update(set(stats["matched_terms"]))

    # 没有有效语料时提前返回，后面的平均长度计算才不会除以 0。
    if not corpus:
        return []

    average_document_length = total_document_length / len(corpus)  # 当前过滤后语料的平均长度，供 BM25 归一化
    top_candidates: list[tuple[float, int, dict[str, Any]]] = []  # 固定容量小顶堆，堆顶是保留结果中分数最低的一项
    # 第二遍遍历基于完整语料统计打分；必须先知道平均长度和文档频率，不能在第一遍随便用局部统计定最终分。
    for sequence, stats in enumerate(corpus):
        doc = stats["doc"]
        score = _bm25_score(
            stats,
            document_count=len(corpus),
            document_frequency=document_frequency,
            average_document_length=average_document_length,
        )
        # 无正向关键词贡献的文档不进入候选堆，让有限位置留给实际命中查询的记录。
        if score <= 0:
            continue
        item = _document_result(doc, score, "keyword", "keyword_score")
        item["bm25_score"] = round(score, 4)
        item["bm25_matched_terms"] = list(stats["matched_terms"])
        item["retrieval"].update(
            {
                "keyword_backend": "bm25",
                "bm25_score": round(score, 4),
                "bm25_matched_terms": list(stats["matched_terms"]),
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
            }
        )
        # sequence 在分数相同时提供可比较次序，避免堆尝试直接比较两个 dict 触发类型错误。
        entry = (score, sequence, item)
        if len(top_candidates) < fetch_limit:
            heapq.heappush(top_candidates, entry)
        # 小顶堆顶是当前最弱保留候选；新分数更高才替换，固定堆容量不会随语料无限增长。
        elif score > top_candidates[0][0]:
            heapq.heapreplace(top_candidates, entry)
    # 堆中只保证堆顶最小，不保证整体已排序；输出前按分数从高到低整理，再归一化和登记排名。
    candidates = [
        entry[2]
        for entry in sorted(top_candidates, key=lambda entry: (entry[0], entry[1]), reverse=True)
    ]
    _normalise_scores(candidates, "keyword_score")
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["keyword_score"]
        item["retrieval"]["keyword_score"] = item["keyword_score"]
        item["keyword_rank"] = rank
        item["retrieval"]["keyword_rank"] = rank
    return candidates


# 先按 chunk_id/id 等稳定键，把同一片段的向量与关键词命中合成一个候选。
# 同一文档从两路进来时保留各自更高分，并合并轨迹来源，不要把它算成两份独立证据。
# 融合公式是 vector_weight*vector_score + keyword_weight*keyword_score + 双路命中奖励，再限制到 1。
# 当前权重为 0.62/0.30，奖励为 0.08；只命中一路的候选没有双路奖励。
# 排序后写 fusion_rank，原路排名/分数仍保留，便于解释为什么最终位置变化。
# 这不是按倒数排名相加的 RRF；面试说明应与实际代码的加权公式一致。
def _merge_candidates(vector_items: list[dict[str, Any]], keyword_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}  # 同一片段合并到同一个键，不重复算证据
    # 外层区分两条召回路径，内层逐条合并候选；这样既能去重，又能保留它究竟被哪一路召回。
    for source_name, items in [("vector", vector_items), ("keyword", keyword_items)]:
        # 同一片段可能已在另一路出现，先用稳定身份查 merged，首次复制，重复则补分数和来源。
        for item in items:
            metadata = item.get("metadata") or {}
            key = str(
                metadata.get("chunk_id")
                or item.get("id")
                or f"{item.get('source_type')}:{item.get('source_id')}:{item.get('title')}"
            )
            current = merged.get(key)
            # 首次见到这份证据时复制 retrieval 字典，避免合并轨迹时顺手改到原始召回结果。
            if current is None:
                current = {**item, "retrieval": {**(item.get("retrieval") or {})}}
                current["retrieval"]["sources"] = list(dict.fromkeys(current["retrieval"].get("sources") or [source_name]))
                merged[key] = current
            else:
                current["vector_score"] = max(float(current.get("vector_score") or 0.0), float(item.get("vector_score") or 0.0))
                current["keyword_score"] = max(float(current.get("keyword_score") or 0.0), float(item.get("keyword_score") or 0.0))
                current["vector_score_raw"] = max(
                    float(current.get("vector_score_raw") or 0.0), float(item.get("vector_score_raw") or 0.0)
                )
                current["bm25_score"] = max(float(current.get("bm25_score") or 0.0), float(item.get("bm25_score") or 0.0))
                current["snippet"] = max([str(current.get("snippet") or ""), str(item.get("snippet") or "")], key=len)
                current["_raw_content"] = max(
                    [str(current.get("_raw_content") or ""), str(item.get("_raw_content") or "")], key=len
                )
                current["retrieval"]["sources"] = list(dict.fromkeys([*(current["retrieval"].get("sources") or []), source_name]))
                # 逐字段合并诊断；sources 已专门去重合并，不能再被这一轮的单路 sources 覆盖。
                for trace_key, trace_value in (item.get("retrieval") or {}).items():
                    if trace_key == "sources":
                        continue
                    current["retrieval"][trace_key] = trace_value
    output = []
    # 两路全部登记后，才对每个唯一片段计算融合分；否则同一片段会拿到多次重复的融合结果。
    for item in merged.values():
        vector_score = float(item.get("vector_score") or 0.0)
        keyword_score = float(item.get("keyword_score") or 0.0)
        # 双路有正分支持才奖励；某一路只是字段存在但分数为零，不应获得双路命中加成。
        both_bonus = DUAL_RECALL_BONUS if vector_score and keyword_score else 0.0
        # 分别乘路权重，再加双路命中奖励，并限制上界；这里只融合检索相关性，不计算答案可信概率。
        combined = min(1.0, VECTOR_FUSION_WEIGHT * vector_score + KEYWORD_FUSION_WEIGHT * keyword_score + both_bonus)
        item["combined_score"] = round(combined, 4)
        item["fusion_score"] = round(combined, 4)
        item["score"] = round(combined, 4)
        item["retrieval"].update(
            {
                "fusion_method": "weighted",
                "fusion_weights": {
                    "vector": VECTOR_FUSION_WEIGHT,
                    "keyword": KEYWORD_FUSION_WEIGHT,
                    "dual_recall_bonus": DUAL_RECALL_BONUS,
                },
                "dual_recall_bonus": both_bonus,
                "combined_score": item["combined_score"],
                "fusion_score": item["fusion_score"],
            }
        )
        output.append(item)
    output.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
    # 最终排序之后才赋融合排名，不能沿用任意一路原排名冒充最终名次。
    for rank, item in enumerate(output, start=1):
        item["fusion_rank"] = rank
        item["retrieval"]["fusion_rank"] = rank
    return output


def _normalised_fingerprint(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", str(text or "").lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest() if normalised else ""


def _candidate_term_set(item: dict[str, Any]) -> set[str]:
    text = str(item.get("_raw_content") or item.get("snippet") or "")
    return set(_tokens(text))


def _same_duplicate_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_meta = left.get("metadata") or {}
    right_meta = right.get("metadata") or {}
    left_parent = str(left_meta.get("parent_id") or "")
    right_parent = str(right_meta.get("parent_id") or "")
    if left_parent and right_parent:
        return left_parent == right_parent
    return (
        str(left.get("source_type") or "") == str(right.get("source_type") or "")
        and str(left.get("source_id") or "") == str(right.get("source_id") or "")
    )


# 对同一来源/父文档内候选先做规范化文本指纹去重，再用词集合 Jaccard 查近重复。
# 规范化移除大小写/部分标点差异；Jaccard 用交集大小除以并集大小，达到阈值才合并。
# 限制 duplicate scope，避免两个不同来源恰好有同样一句话就被当成完全相同证据丢掉。
# 按已有排序遍历，所以较早保留的候选作为代表，后来的重复项在 trace 中记录 duplicate_of。
# 输出保留列表与删除清单，让学员能排查 Top-K 变少是否只是去除了重复。
def _deduplicate_candidates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []  # 按当前排名保留的代表候选；初始还没有接收任何候选
    removed: list[dict[str, Any]] = []  # 被判重复的候选与其代表身份，供诊断解释数量为什么减少
    fingerprints: dict[str, dict[str, Any]] = {}  # 规范化内容指纹到已保留代表片段的映射
    term_sets: dict[str, set[str]] = {}  # 缓存已检查候选的词集合，后续相似度比较尽量不重复分词
    # 按已有高分顺序逐条决定留删，先到的代表保留，后到的重复内容不再占 Top-K 位置。
    for item in items:
        item_id = str(item.get("id") or "")
        raw_text = str(item.get("_raw_content") or item.get("snippet") or "")
        fingerprint = _normalised_fingerprint(raw_text)
        duplicate_of: dict[str, Any] | None = fingerprints.get(fingerprint) if fingerprint else None
        if duplicate_of is not None and not _same_duplicate_scope(item, duplicate_of):
            duplicate_of = None
        if duplicate_of is None:
            current_terms = _candidate_term_set(item)
            # 完全指纹未命中时，再与已保留的同范围候选比较近重复，不能拿不同来源的相似措辞随意合并。
            for existing in kept:
                if not _same_duplicate_scope(item, existing):
                    continue
                existing_id = str(existing.get("id") or "")
                existing_terms = term_sets.get(existing_id) or _candidate_term_set(existing)
                if not current_terms or not existing_terms:
                    continue
                # 并集做分母使相似度落在 0..1；不同长度文本也可比较，但词集合会忽略重复词次。
                union = current_terms | existing_terms
                similarity = len(current_terms & existing_terms) / max(len(union), 1)
                if similarity >= NEAR_DUPLICATE_JACCARD:
                    duplicate_of = existing
                    break
            term_sets[item_id] = current_terms
        # 已确认重复时把删除关系记到代表和 removed，再跳过加入 kept；保留的是审计线索而非第二份正文。
        if duplicate_of is not None:
            duplicate_id = str(duplicate_of.get("id") or "")
            duplicate_trace = duplicate_of.setdefault("retrieval", {})
            duplicate_trace.setdefault("duplicates_removed", []).append(item_id)
            removed.append({"id": item_id, "duplicate_of": duplicate_id})
            continue
        kept.append(item)
        if fingerprint:
            fingerprints[fingerprint] = item
    return kept, removed


# 限制同一个 parent_id 最多占 MAX_CHUNKS_PER_PARENT 个位置，提高结果来源多样性。
# 例如一本长手册有十个相似切片高分，也应为其他相关资料留出候选位置。
# 它与文本去重不同：两个内容不同的同父切片也可能因占位上限被移除。
# 没有 parent_id 的兼容候选不按这个计数器聚合；移除清单会保存在流水线轨迹中。
def _limit_parent_occupancy(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()  # 按 parent_id 统计已占用候选位置
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    # 按排序逐条分配父文档名额，优先让高分片段占位；不是先按父文档分组打乱排名。
    for item in items:
        metadata = item.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id and counts[parent_id] >= MAX_CHUNKS_PER_PARENT:
            removed.append({"id": str(item.get("id") or ""), "parent_id": parent_id})
            continue
        kept.append(item)
        if parent_id:
            # 只有真正保留的片段才消耗名额，被移除的候选不再计数。
            counts[parent_id] += 1
    return kept, removed


# 比较 branch/commit_sha/version/checksum/knowledge_document_id 等共同存在的版本线索。
# 双方同一字段都有值且不相同就拒绝扩展，防止同父附近的旧版片段混到新版证据里。
# 缺失字段不会在此自动视为冲突，因此这是对已有元数据的兼容检查，而非完整版本证明。
def _same_version(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # 逐个比较双方都有的版本字段，一项明确冲突就不能拼接；字段缺失本身不被此实现判为冲突。
    for key in ("branch", "commit_sha", "version", "checksum", "knowledge_document_id"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    return True


# 相关子片太短时，补回同父下相邻 child_index 的片段，以恢复局部阅读语境。
# 先按仓库、source_type、source_id 查询兄弟，再检查 parent_id 和共同版本信息。
# neighbor_window=1 表示最多考虑左右各一个索引邻居，不是把整篇父文档无上限拼接。
# 按 child_index 排序并规范化指纹去重后拼正文，保留扩展的 chunk IDs 与原检索分数。
# 缺父信息、非法来源 UUID 或找不到匹配兄弟时保留原候选，并在 trace 解释没有扩展的原因。
# 后面仍要做总 token 压缩，扩展成功不意味着模型最终看到所有拼接字符。
def expand_parent_context(
    db: Session,
    repo_id: str | uuid.UUID,
    items: list[dict[str, Any]],
    *,
    neighbor_window: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: list[dict[str, Any]] = []  # 返回候选列表，包括扩展成功项及无法扩展而原样保留的项
    expanded_items = 0  # 真正补入邻居的候选条数；没有处理候选之前为 0
    expanded_chunks = 0  # 累计额外补入的片段数，与“扩展了几条候选”是两个统计
    # 每条命中分别尝试恢复局部语境，缺定位或无邻居时仍保留该候选，不能让扩展失败等于丢失召回。
    for item in items:
        metadata = dict(item.get("metadata") or {})
        parent_id = str(metadata.get("parent_id") or "")
        child_index = metadata.get("child_index")
        source_id = item.get("source_id")
        source_type = str(item.get("source_type") or "")
        retrieval = dict(item.get("retrieval") or {})
        if not parent_id or child_index is None or not source_id or not source_type:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "parent metadata unavailable"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        try:
            source_uuid = _as_uuid(source_id)
        except (TypeError, ValueError):
            retrieval["parent_expansion"] = {"expanded": False, "reason": "invalid source id"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        siblings = (
            db.query(Document)
            .filter(
                Document.repo_id == _as_uuid(repo_id),
                Document.source_type == source_type,
                Document.source_id == source_uuid,
            )
            .all()
        )
        selected: list[Document] = []
        current_index = int(child_index)  # 当前命中子片在父文档内的位置，用于判断邻居距离
        # 先筛同父与兼容版本，再按 child_index 距离选择左右邻居；同一 source_id 下不一定全属同一父区段。
        for sibling in siblings:
            sibling_meta = sibling.meta or {}
            if str(sibling_meta.get("parent_id") or "") != parent_id:
                continue
            # 同父 ID 不足以放行：重新上传或切换分支后可能仍有相近定位，需继续核对共同版本字段。
            if not _same_version(metadata, sibling_meta):
                continue
            sibling_index = int(sibling_meta.get("child_index") or 0)
            if abs(sibling_index - current_index) <= max(0, neighbor_window):
                selected.append(sibling)
        # 相邻片段必须按文内次序拼接，不能按数据库返回顺序或相似度顺序打乱原文。
        selected.sort(key=lambda row: int((row.meta or {}).get("child_index") or 0))
        if not selected:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "no matching siblings"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parts: list[str] = []
        seen_parts: set[str] = set()
        expanded_ids: list[str] = []
        # 逐片清理空正文和重复内容，同时保存实际采用的 chunk ID，扩展结果才能回查。
        for sibling in selected:
            content = str(sibling.content or "").strip()
            fingerprint = _normalised_fingerprint(content)
            if not content or fingerprint in seen_parts:
                continue
            seen_parts.add(fingerprint)
            parts.append(content)
            expanded_ids.append(str(sibling.id))
        if not parts:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "empty sibling content"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parent_expanded = len(expanded_ids) > 1 or str(item.get("id") or "") not in expanded_ids
        retrieval["parent_expansion"] = {
            "expanded": parent_expanded,
            "parent_id": parent_id,
            "chunk_ids": expanded_ids,
            "neighbor_window": max(0, neighbor_window),
        }
        next_metadata = {**metadata, "expanded_chunk_ids": expanded_ids}
        expanded.append(
            {
                **item,
                "metadata": next_metadata,
                "_raw_content": "\n\n".join(parts),
                "parent_expanded": parent_expanded,
                "retrieval": retrieval,
            }
        )
        if parent_expanded:
            expanded_items += 1
            # 减去原本命中的那一片，只统计额外片数，避免把原片也算作新增上下文。
            expanded_chunks += max(0, len(expanded_ids) - 1)
    return expanded, {
        "name": "parent_expansion",
        "input_count": len(items),
        "output_count": len(expanded),
        "expanded_items": expanded_items,
        "additional_chunks": expanded_chunks,
    }


# 通用召回入口，参数决定向量/关键词/混合路径、来源范围、是否重排与是否压缩。
# 先过滤不允许搜索的来源，再把候选池放大到目标数量数倍，并设置硬上限控制成本。
# 向量、BM25 各保留自己的原分和排名；混合时融合后去重并限制同父占位，再选择最终候选。
# Milvus 不可用时，hybrid 可以保留关键词召回，但 vector 请求必须抛出错误，不能偷换检索方式。
# 上层知识库编排可能关闭此处重排/压缩，在它自己的外部重排、父扩展和阈值之后再处理。
# pipeline_trace 逐段记录数量与模式，排查时可区分“没召回”“去重减少”“阈值过滤”或“服务降级”。
# 返回的是候选证据，不在这里生成回答或证明候选事实仍是最新状态。
def search_similar_documents(
    db: Session,  # 关键词分支读取关系库文档，向量分支使用 Milvus
    repo_id: str | uuid.UUID,  # 当前仓库隔离范围，不应跨项目召回
    query: str,  # 本次实际查询，可以与用户原问题不同
    source_type: str | None = None,  # 单一来源筛选的兼容参数
    limit: int = 5,  # 目标结果数，内部先放大候选池
    metadata_filters: dict | None = None,  # 状态、路径、标签与时间等条件
    query_embedding: list[float] | None = None,  # 可复用上层已生成的查询向量
    retrieval_method: str = "hybrid",  # vector/fulltext/hybrid 分支选择
    apply_rerank: bool = True,  # 上层要外部重排时可先关闭这里
    apply_compression: bool = True,  # 是否在此处裁成提示片段
    source_types: list[str] | None = None,  # 多来源筛选，仍受来源策略约束
    pipeline_trace: dict[str, Any] | None = None,  # 调用方提供的诊断字典，会写入阶段信息
) -> list[dict]:
    requested_types = normalize_source_types([source_type, *(source_types or [])]) or sorted(RAG_SOURCE_TYPES)
    searchable_types = list(dict.fromkeys(item for item in requested_types if is_searchable_source(item)))
    if not searchable_types:
        return []

    # 先扩大候选池给融合/去重/重排留余量，再守住上限；Top-K 不是每个阶段都只保留 K 条。
    fetch_limit = min(max(limit * 8, 40), 200)
    method = retrieval_method.strip().lower()
    vector_types = searchable_types
    # 向量和关键词结果分开收集；空列表表示该路尚未返回候选，不表示服务必然失败。
    vector_candidates: list[dict[str, Any]] = []
    # 关键词支路独立运行，hybrid 降级时可保留它的真实结果，但不能冒充向量命中。
    keyword_candidates: list[dict[str, Any]] = []
    # 本次检索的阶段记录从空开始，每步追加输入输出数量，便于定位结果在哪一步变少。
    stages: list[dict[str, Any]] = []
    vector_status = "skipped"  # 区别未选择向量分支与实际请求失败
    if method in {"vector", "hybrid"} and vector_types:
        try:
            vector_candidates = _milvus_vector_candidates(
                repo_id,
                query,
                vector_types,
                fetch_limit,
                metadata_filters,
                query_embedding=query_embedding,
            )
            vector_status = "success"
        # hybrid 允许继续走关键词但保留不可用状态；vector 模式没有等价替代分支，必须继续抛错。
        except MilvusUnavailableError:
            vector_status = "unavailable"
            if method == "vector":
                raise
    stages.append(
        {
            "name": "vector_recall",
            "status": vector_status,
            "candidate_count": len(vector_candidates),
            "backend": "milvus",
        }
    )
    keyword_types = searchable_types if method in {"fulltext", "hybrid"} else []
    if keyword_types:
        keyword_candidates = _keyword_candidates(
            db,
            repo_id,
            query,
            keyword_types,
            fetch_limit,
            metadata_filters,
        )
    stages.append(
        {
            "name": "keyword_recall",
            "status": "success" if keyword_types else "skipped",
            "candidate_count": len(keyword_candidates),
            "backend": "bm25",
        }
    )
    if method == "vector":
        merged = vector_candidates
    elif method == "fulltext":
        merged = keyword_candidates
    else:
        merged = _merge_candidates(vector_candidates, keyword_candidates)
    stages.append(
        {
            "name": "weighted_fusion" if method == "hybrid" else "single_retriever_selection",
            "method": "weighted" if method == "hybrid" else method,
            "candidate_count": len(merged),
            "weights": {
                "vector": VECTOR_FUSION_WEIGHT,
                "keyword": KEYWORD_FUSION_WEIGHT,
                "dual_recall_bonus": DUAL_RECALL_BONUS,
            },
        }
    )
    # 先去重复，再限制单父文档占位，分别解决重复噪声与来源集中问题。
    deduplicated, duplicates_removed = _deduplicate_candidates(merged)
    aggregated, parent_overflow_removed = _limit_parent_occupancy(deduplicated)
    stages.append(
        {
            "name": "deduplication_and_parent_aggregation",
            "input_count": len(merged),
            "output_count": len(aggregated),
            "duplicates_removed": duplicates_removed,
            "parent_overflow_removed": parent_overflow_removed,
        }
    )
    ranked = rerank_documents(query, aggregated, limit) if apply_rerank else aggregated[:limit]
    if apply_rerank:
        stages.append(
            {
                "name": "rerank",
                "input_count": len(aggregated),
                "output_count": len(ranked),
                "mode": ranked[0].get("retrieval", {}).get("rerank_mode") if ranked else None,
            }
        )
    if pipeline_trace is not None:
        pipeline_trace.update(
            {
                "retrieval_method": method,
                "fusion_method": "weighted" if method == "hybrid" else method,
                "stages": stages,
            }
        )
    # 上层要对原文做外部重排/父扩展时，从这里拿未压缩结果，避免重要判断句已被先截掉。
    if not apply_compression:
        return ranked
    compressed = compress_retrieval_results(query, ranked, max_tokens_per_item=260)
    if pipeline_trace is not None:
        pipeline_trace["stages"].append(
            {
                "name": "context_compression",
                "input_count": len(ranked),
                "output_count": len(compressed),
            }
        )
    return compressed
