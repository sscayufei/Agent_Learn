# 轻量启发式重排：在候选召回之后综合查询词、标题/正文等信号调整顺序。
# 重排不负责全库召回；上游完全没召回的文档，靠这一步无法凭空找回来。
import math
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _normalise(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _candidate_text(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "")
    return "\n".join(
        [
            str(candidate.get("title") or ""),
            path,
            str(candidate.get("snippet") or ""),
        ]
    ).lower()


def _freshness_bonus(metadata: dict[str, Any]) -> float:
    # 按字段优先顺序找第一个可用时间，解析失败才继续下一个，不把多个时间加成重复相加。
    for key in ("updated_at", "created_at", "merged_at", "closed_at"):
        value = metadata.get(key)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        # 统一时区后计算距离今天的天数；未来时间按 0 天处理，避免产生异常负年龄。
        age_days = max(0, (datetime.now(timezone.utc) - parsed).days)
        if age_days <= 30:
            return 0.04
        if age_days <= 180:
            return 0.02
        return 0.0
    return 0.0


# 综合查询词覆盖、候选标题/内容和已有检索分等线索，计算可重复的本地重排分。
# 新鲜度等加成只作为排序信号，不能由此证明某份资料一定是当前唯一权威版本。
# 这一路无需额外大模型，适合回退和工程测试；它与外部交叉编码/重排模型的能力不同。
def _heuristic_score(query: str, candidate: dict[str, Any]) -> tuple[float, str]:
    terms = _tokens(query)
    if not terms:
        return _normalise(float(candidate.get("score") or 0.0)), "base score only"
    text = _candidate_text(candidate)
    title = str(candidate.get("title") or "").lower()
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "").lower()
    matched = [term for term in terms if term in text]
    coverage = len(set(matched)) / max(len(set(terms)), 1)  # 命中不同查询词的比例；分子分母都去重，分母至少为 1
    title_hits = sum(1 for term in terms if term in title)
    path_hits = sum(1 for term in terms if term in path)
    phrase_bonus = 0.12 if query.lower().strip() and query.lower().strip() in text else 0.0
    source_bonus = 0.08 if len({term for term in terms if term in text}) >= 2 else 0.0
    base = _normalise(float(candidate.get("score") or 0.0))
    keyword = _normalise(float(candidate.get("keyword_score") or 0.0))
    vector = _normalise(float(candidate.get("vector_score") or 0.0))
    sources = set((candidate.get("retrieval") or {}).get("sources") or [])
    dual_recall_bonus = 0.05 if {"vector", "keyword"}.issubset(sources) else 0.0
    freshness_bonus = _freshness_bonus(metadata)
    source_type = str(candidate.get("source_type") or "").lower()
    source_type_bonus = 0.03 if source_type and any(term in source_type for term in terms) else 0.0
    # 多种弱排序线索按当前固定权重相加；标题/路径命中有上限，避免重复关键词无限加分。
    score = (
        0.31 * base
        + 0.20 * vector
        + 0.16 * keyword
        + 0.17 * coverage
        + 0.04 * min(title_hits, 3) / 3
        + 0.02 * min(path_hits, 3) / 3
        + phrase_bonus
        + source_bonus
        + dual_recall_bonus
        + freshness_bonus
        + source_type_bonus
    )
    reasons = []  # 逐项收集实际生效的排序理由，不凭空给候选贴“高可信”标签
    if coverage:
        reasons.append(f"matched {len(set(matched))}/{len(set(terms))} query terms")
    if title_hits:
        reasons.append("title match")
    if path_hits:
        reasons.append("path match")
    if vector:
        reasons.append("vector candidate")
    if keyword:
        reasons.append("keyword candidate")
    if dual_recall_bonus:
        reasons.append("dual recall")
    if freshness_bonus:
        reasons.append("fresh evidence")
    if source_type_bonus:
        reasons.append("source type match")
    return _normalise(score), ", ".join(reasons) or "弱关键词匹配"


# 对已召回候选计算启发式分数并排序，返回最多 limit 条，同时保留重排模式与分数轨迹。
# 排序后的分与原向量/BM25 分分别记录，使调试时能解释候选为什么上升/下降。
# 没有进入输入候选的资料不可能被此处找出来，召回不足要回头查上游过滤、索引与查询。
def rerank_documents(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    # 至少覆盖最终需要的条数，但不超过已有候选数量；只在头部候选花重排计算。
    top_n = min(max(settings.rag_rerank_top_n, limit), len(candidates))
    head = candidates[:top_n]
    tail = candidates[top_n:]
    ranked: list[dict[str, Any]] = []
    # 头部每条都按同一公式重算并记录理由，尾部暂保原序，召回列表之外的资料不会在这里突然出现。
    for candidate in head:
        heuristic_score, heuristic_reason = _heuristic_score(query, candidate)
        ranked.append(
            {
                **candidate,
                "score": round(heuristic_score, 4),
                "rerank_score": round(heuristic_score, 4),
                "rank_reason": heuristic_reason,
                "retrieval": {
                    **(candidate.get("retrieval") or {}),
                    "rerank_mode": "heuristic",
                    "rerank_score": round(heuristic_score, 4),
                    "rank_reason": heuristic_reason,
                },
            }
        )
    ranked.sort(key=lambda item: (float(item.get("rerank_score") or item.get("score") or 0.0), float(item.get("combined_score") or 0.0)), reverse=True)
    # 重排头部在前、未重排尾部在后，最终再截到 limit；不是无限返回全部候选。
    return [*ranked, *tail][:limit]
