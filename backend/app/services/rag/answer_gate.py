# 生成前的回答门控：无证据、分数不足、关键名称未命中、证据冲突或问题不明确时，不直接让模型下结论。
# 冲突/歧义检测是词面启发式保护，不是完整事实验证；通过门控也不代表回答已被证明正确。
import re
from typing import Any


AMBIGUOUS_MARKERS = ("这个", "那个", "它", "这段", "这里", "怎么办", "怎么处理")
CONFLICT_POLARITIES = (
    ("permission", ("允许", "可以", "enabled", "enable"), ("禁止", "不允许", "不能", "disabled", "disable")),
    ("outcome", ("成功", "通过", "success", "passed"), ("失败", "未通过", "failure", "failed")),
    ("requirement", ("必须", "需要", "required"), ("可选", "无需", "optional")),
)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _evidence_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return _normalise(
        "\n".join(
            [
                str(item.get("title") or ""),
                str(item.get("snippet") or item.get("_raw_content") or ""),
                str(metadata.get("path") or ""),
                str(metadata.get("symbol") or ""),
                str(metadata.get("branch") or metadata.get("base_branch") or ""),
            ]
        )
    )


# 抽出文件名、错误名、含数字标识等较强查询线索，供门控核对候选是否真正谈到该对象。
# 自然语言泛词相关不够：用户问某个具体 Error，而候选连名字都没有出现，可能只是话题相近。
# 当前只用词面模式识别，缩写/别名等仍可能漏识别，需要结合人工调试检索。
def _strong_query_signals(query: str) -> list[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_./\-]*|#\d+|\b\d{3,}\b", query)
    signals: list[str] = []
    for candidate in candidates:
        if (
            any(marker in candidate for marker in ("_", ".", "/", "-"))
            or any(char.isdigit() for char in candidate)
            or candidate.endswith(("Error", "Exception"))
        ):
            signals.append(candidate.lower())
    return list(dict.fromkeys(signals))


# 只在较高分候选间检查许可/结果/要求等正反词面信号，返回涉及来源的 ID。
# 例如一份资料写“允许”，另一份写“禁止”，可能存在版本差异或冲突，需要先核对来源。
# 这是防御性启发式，不做完整逻辑推理；同一主题出现正反词不一定总是真矛盾。
# 因此返回 conflict 后应澄清/核验，而不是由程序任意挑一边作为事实。
def _detect_conflict(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    high_confidence = [item for item in items if float(item.get("score") or 0.0) >= 0.6]
    if len(high_confidence) < 2:
        return None
    texts = [(str(item.get("id") or item.get("source_id") or ""), _evidence_text(item)) for item in high_confidence]
    for conflict_type, positive_markers, negative_markers in CONFLICT_POLARITIES:
        negative_ids = [item_id for item_id, text in texts if any(marker in text for marker in negative_markers)]
        positive_ids = [
            item_id
            for item_id, text in texts
            if any(marker in text for marker in positive_markers)
            and not any(marker in text for marker in negative_markers)
        ]
        if positive_ids and negative_ids and set(positive_ids) != set(negative_ids):
            return {
                "type": conflict_type,
                "positive_evidence_ids": positive_ids,
                "negative_evidence_ids": negative_ids,
            }
    return None


def _is_ambiguous_query(query: str, items: list[dict[str, Any]]) -> bool:
    compact = re.sub(r"\s+", "", query)
    if not any(marker in compact for marker in AMBIGUOUS_MARKERS):
        return False
    source_ids = {str(item.get("source_id") or "") for item in items if item.get("source_id")}
    return len(source_ids) > 1 or len(compact) <= 8


# 回答前按顺序检查证据条件，返回结构化 decision，而不是直接写领域答案。
# 先判空和最高分阈值；再确认查询关键标识并非全部缺失；之后看高分冲突和对象范围歧义。
# 资料不足 -> insufficient_evidence；相反高分线索 -> conflict；指代不明 -> ask_clarification。
# 只有这些检查通过才返回 answer，并带匹配信号与分数等诊断供 QA/前端共同使用。
# 检查顺序影响原因：空证据不应被误报成问题歧义，标识完全不符也不该因高分就放行。
# 通过门控只是允许生成，不是事实已被严格证明，更不保证模型最终按要求引用。
def evaluate_answer_gate(
    query: str,
    items: list[dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    threshold_enabled: bool = True,
) -> dict[str, Any]:
    top_score = max((float(item.get("score") or 0.0) for item in items), default=0.0)
    base = {
        "evidence_count": len(items),
        "top_score": round(top_score, 4),
        "score_threshold": round(float(score_threshold), 4) if threshold_enabled else None,
    }
    if not items:
        return {**base, "decision": "insufficient_evidence", "reason": "没有检索到满足条件的证据"}
    if threshold_enabled and top_score < score_threshold:
        return {**base, "decision": "insufficient_evidence", "reason": "最高证据分数未达到回答门槛"}

    signals = _strong_query_signals(query)
    evidence_text = "\n".join(_evidence_text(item) for item in items)
    missing_signals = [signal for signal in signals if signal not in evidence_text]
    if signals and len(missing_signals) == len(signals):
        return {
            **base,
            "decision": "insufficient_evidence",
            "reason": "查询中的关键名称没有出现在候选证据中",
            "required_signals": signals,
            "missing_signals": missing_signals,
        }

    conflict = _detect_conflict(items)
    if conflict:
        return {**base, "decision": "conflict", "reason": "高排名证据给出了相反结论", "conflict": conflict}

    if _is_ambiguous_query(query, items):
        return {
            **base,
            "decision": "ask_clarification",
            "reason": "问题缺少可定位的对象或范围",
            "suggested_fields": ["文件或模块", "错误名称", "分支或版本"],
        }

    return {
        **base,
        "decision": "answer",
        "reason": "证据达到分数门槛，关键名称可以在候选中核对",
        "matched_signals": [signal for signal in signals if signal in evidence_text],
    }


def gate_response_message(gate: dict[str, Any]) -> str:
    decision = str(gate.get("decision") or "insufficient_evidence")
    if decision == "ask_clarification":
        return "当前问题的对象或范围不够明确。请补充文件、模块、错误名称、分支或版本后再试。"
    if decision == "conflict":
        return "当前知识库中的高排名证据存在冲突，暂时无法给出确定结论。请先核对对应来源或版本。"
    return "当前知识库资料不足，无法确认。请补充相关资料、检查检索范围，或换一种更具体的问法。"
