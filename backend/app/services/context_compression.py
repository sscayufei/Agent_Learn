# 上下文预算与压缩的纯逻辑层：估算 token、保留调用配对、裁剪证据、生成摘要和合并记忆。
# 估算用于保护输入窗口，不是计费统计；压缩用于控制模型输入，不等于删除数据库历史。
# 关键约束：保留用户任务、事实与引用，保留工具调用/结果配对，并给输出留足空间。
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import settings


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_\-]+|[^\s]")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?\n\u3002\uff01\uff1f])\s+")

MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
}

MODEL_OUTPUT_WINDOWS = {
    "deepseek-v4-pro": 384_000,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4.1": 32_768,
    "gpt-4.1-mini": 32_768,
    "gpt-4.1-nano": 32_768,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "o1": 100_000,
    "o1-mini": 65_536,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
    "claude-3-5-sonnet": 8_192,
    "claude-3-7-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-opus-4": 64_000,
}

COMPACTABLE_TOOL_HINTS = (
    "workspace.read_file",
    "workspace.search_code",
    "workspace.list_files",
    "rag.search_similar_documents",
    "devflow_search_evidence",
    "repo_health.generate_summary",
    "ci_debug_agent.analyze_workflow_run",
    "read",
    "grep",
    "search",
    "shell",
    "web_fetch",
    "webfetch",
)

COMPACT_BOUNDARY_SUBTYPE = "compact_boundary"
MICROCOMPACT_MARKER = "[Old regenerable tool/context result cleared by microcompact]"


PressureStage = Literal["normal", "warning", "auto_compact", "aggressive", "manual_path"]


# 用字符特征近似估算中英文混合文本的 token 消耗，便于调用前快速分预算。
# 中文、英文单词和标点的实际分词受模型 tokenizer 影响，因此这不是精确计费值。
# 估算会贯穿历史、证据和工具观察；供应商返回的真实 usage 另行记录用于校准诊断。
# 安全预留和单次溢出恢复共同弥补近似误差，不能把估算恰好贴窗口当作可靠输入上限。
def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    total = 0
    for token in TOKEN_PATTERN.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            total += 1
        elif re.fullmatch(r"[a-zA-Z0-9_\-]+", token):
            total += max(1, (len(token) + 3) // 4)
        else:
            total += 1
    return total


def is_context_overflow_error(exc: Exception) -> bool:
    """Recognize provider-specific prompt/context overflow failures.

    OpenAI-compatible gateways do not expose one stable exception type, so we
    deliberately use a narrow set of message markers and only allow callers to
    retry once.
    """
    message = str(exc or "").lower()
    markers = (
        "context_length_exceeded",
        "maximum context length",
        "prompt is too long",
        "prompt too long",
        "context window",
        "too many tokens",
        "request too large",
    )
    return any(marker in message for marker in markers)


def _normalize_model_name(model: str | None) -> str:
    value = str(model or settings.llm_model or "").strip().lower()
    if not value:
        return "unknown"
    value = value.split("/")[-1]
    value = re.sub(r"-(20\d{2}[-_]\d{2}[-_]\d{2}|\d{4}[-_]\d{2}[-_]\d{2})$", "", value)
    return value


def _model_lookup(model: str | None, table: dict[str, int]) -> int | None:
    normalized = _normalize_model_name(model)
    if normalized in table:
        return table[normalized]
    for key, value in table.items():
        if normalized.startswith(key):
            return value
    return None


def model_context_window_tokens(model: str | None = None) -> int:
    if settings.context_model_window_tokens:
        return max(4096, settings.context_model_window_tokens)
    known = _model_lookup(model, MODEL_CONTEXT_WINDOWS)
    if known:
        return known
    return max(4096, settings.context_max_input_tokens + settings.context_reserved_response_tokens)


def model_output_window_tokens(model: str | None = None) -> int:
    known = _model_lookup(model, MODEL_OUTPUT_WINDOWS)
    if known:
        return known
    return max(512, settings.context_reserved_response_tokens)


# 把模型支持窗口、项目允许输入和摘要场景预留等约束合成有效窗口。
# 模型理论窗口很大，也不意味着每次请求应无条件填满；响应还要占输出空间。
# for_compaction 区分摘要请求与普通回答，避免二者拿错输出预留。
# 后续压力阶段应以这个有效值为基准，而非只看某个静态模型名称对应的标称上限。
def effective_context_window_tokens(model: str | None = None, *, for_compaction: bool = False) -> int:
    model_window = model_context_window_tokens(model)
    if for_compaction:
        reserved = min(model_output_window_tokens(model), max(512, settings.context_compact_reserved_output_tokens))
    else:
        reserved = max(512, min(settings.context_reserved_response_tokens, model_window // 2))
    dynamic_window = max(4096, model_window - reserved)
    if settings.context_dynamic_model_window_enabled:
        return max(settings.context_max_input_tokens, dynamic_window)
    return max(4096, min(settings.context_max_input_tokens, dynamic_window))


@dataclass(frozen=True)
class ContextPressure:
    stage: PressureStage  # 当前压力阶段，决定提示、微压缩或正式摘要路径
    token_usage: int  # 当前输入的估算用量，不是计费用量
    effective_window_tokens: int  # 综合模型与配置后采用的窗口额度
    remaining_tokens: int  # 估算剩余空间，用于判断是否需要留出余量
    warning_threshold: int  # 进入预警阶段的用量阈值
    auto_compact_threshold: int  # 触发自动压缩的用量阈值
    aggressive_threshold: int  # 进入更紧压缩策略的用量阈值
    manual_threshold: int  # 进入极限恢复路径的用量阈值
    utilization: float  # 估算用量占有效窗口的比例

    @property
    def should_warn(self) -> bool:
        return self.stage != "normal"

    @property
    def should_auto_compact(self) -> bool:
        return self.stage in {"auto_compact", "aggressive", "manual_path"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "token_usage": self.token_usage,
            "effective_window_tokens": self.effective_window_tokens,
            "remaining_tokens": self.remaining_tokens,
            "warning_threshold": self.warning_threshold,
            "auto_compact_threshold": self.auto_compact_threshold,
            "aggressive_threshold": self.aggressive_threshold,
            "manual_threshold": self.manual_threshold,
            "utilization": self.utilization,
        }


def _thresholds_for_window(window: int) -> tuple[int, int, int, int]:
    # 阈值前要预留缓冲，且小窗口不能照搬一个过大的固定预留；按窗口比例收紧上限。
    warning_buffer = min(settings.context_warning_buffer_tokens, max(512, window // 4))
    auto_buffer = min(settings.context_auto_compact_buffer_tokens, max(512, window // 5))
    aggressive_buffer = min(settings.context_aggressive_buffer_tokens, max(512, window // 8))
    manual_buffer = min(settings.context_manual_buffer_tokens, max(256, window // 16))
    # 从窗口减去预留得到触发用量，而不是把预留本身当阈值；下界防止极小配置过早触发。
    auto_threshold = max(1024, window - auto_buffer)
    warning_threshold = max(512, min(auto_threshold - 1, auto_threshold - warning_buffer))
    # 用 +1 确保更强阶段严格晚于普通自动压缩，避免多个阶段在同一阈值重叠。
    aggressive_threshold = max(auto_threshold + 1, window - aggressive_buffer)
    manual_threshold = max(aggressive_threshold + 1, window - manual_buffer)
    return warning_threshold, auto_threshold, aggressive_threshold, manual_threshold


# 将当前估算用量与有效窗口比较，计算剩余 headroom 和对应压力阶段。
# warning 用于提示与微压缩，auto_compact/aggressive/manual_path 对应逐步更紧的恢复策略。
# 阶段判断与 configured buffer 共同决定；返回对象携带阈值与窗口，界面可解释为何触发。
# 这是资源压力评估，不是回答质量评分；压力低不能说明当前证据足够。
def calculate_context_pressure(token_usage: int, model: str | None = None) -> ContextPressure:
    effective_window = effective_context_window_tokens(model, for_compaction=True)
    warning_threshold, auto_threshold, aggressive_threshold, manual_threshold = _thresholds_for_window(effective_window)
    # 按压力从高到低判断，因为达到最高阈值也必然超过较低阈值；反过来会总被低级分支抢先命中。
    if token_usage >= manual_threshold:
        stage: PressureStage = "manual_path"
    elif token_usage >= aggressive_threshold:
        stage = "aggressive"
    elif token_usage >= auto_threshold:
        stage = "auto_compact"
    elif token_usage >= warning_threshold:
        stage = "warning"
    else:
        stage = "normal"
    return ContextPressure(
        stage=stage,
        token_usage=token_usage,
        effective_window_tokens=effective_window,
        remaining_tokens=max(0, effective_window - token_usage),
        warning_threshold=warning_threshold,
        auto_compact_threshold=auto_threshold,
        aggressive_threshold=aggressive_threshold,
        manual_threshold=manual_threshold,
        utilization=round(token_usage / effective_window, 4) if effective_window else 0.0,
    )


def is_compact_boundary_message(message: dict[str, Any]) -> bool:
    meta = message.get("meta") or message.get("metadata") or {}
    return str(message.get("role") or "") == "system" and meta.get("subtype") == COMPACT_BOUNDARY_SUBTYPE


def find_last_compact_boundary_index(messages: list[dict[str, Any]]) -> int:
    # 从末尾向前找，遇到的第一个摘要边界就是最新边界；旧边界不应覆盖新摘要。
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[index]):
            return index
    # -1 是“未找到”的标记，不是某条有效消息索引；调用方先检查再做切片。
    return -1


def messages_after_compact_boundary(messages: list[dict[str, Any]], *, include_boundary: bool = True) -> list[dict[str, Any]]:
    boundary_index = find_last_compact_boundary_index(messages)
    if boundary_index < 0:
        return messages
    start = boundary_index if include_boundary else boundary_index + 1
    return messages[start:]


def _message_meta(message: dict[str, Any]) -> dict[str, Any]:
    meta = message.get("meta") or message.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _message_provider_id(message: dict[str, Any]) -> str | None:
    meta = _message_meta(message)
    value = (
        message.get("message_id")
        or message.get("provider_message_id")
        or meta.get("message_id")
        or meta.get("provider_message_id")
    )
    return str(value) if value else None


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _tool_use_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            value = (
                call.get("id")
                or call.get("tool_call_id")
                or call.get("tool_use_id")
                or call.get("call_id")
            )
            if value:
                ids.add(str(value))
    for block in _content_blocks(message):
        if block.get("type") == "tool_use" and block.get("id"):
            ids.add(str(block["id"]))
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and str(message.get("role") or "") in {"assistant", "tool_use"}:
            ids.add(str(meta[key]))
    return ids


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    role = str(message.get("role") or "")
    for key in ("tool_call_id", "tool_use_id"):
        value = message.get(key)
        if value:
            ids.add(str(value))
    for block in _content_blocks(message):
        if block.get("type") == "tool_result" and block.get("tool_use_id"):
            ids.add(str(block["tool_use_id"]))
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and role in {"tool", "tool_result", "user"}:
            ids.add(str(meta[key]))
    return ids


# 从 start_index 开始保留历史后缀之前，修正可能破坏工具协议的截断点。
# 先收集保留段的 tool_result IDs 与 tool_use IDs，差集就是缺少调用的孤立结果。
# 向前搜索对应调用并移动起点，让每个保留结果能找到它的调用声明。
# 随后继续保留同 provider message id 的 assistant 分片，避免拆断供应商的一条逻辑响应。
# 该修正可能让保留数量多于原计划；协议正确性优先于机械的消息条数限制。
def adjust_keep_start_to_preserve_api_invariants(messages: list[dict[str, Any]], start_index: int) -> int:
    """Move a suffix boundary left when slicing there would orphan tool results."""
    if start_index <= 0 or start_index >= len(messages):
        return max(0, min(start_index, len(messages)))

    adjusted = start_index  # 可能向前修正的保留起点，不能直接截出孤立工具结果
    kept_results: set[str] = set()  # 保留区出现的工具结果调用 ID
    kept_uses: set[str] = set()  # 保留区出现的工具调用 ID
    # 先扫计划保留的后缀，收集其中的调用与结果身份，才能知道是否留了结果却删了调用。
    for message in messages[adjusted:]:
        kept_results.update(_tool_result_ids(message))
        kept_uses.update(_tool_use_ids(message))

    # 保留段中的结果减去调用，得到孤立结果的调用 ID；向前找对应 tool_use 来修正切分起点。
    missing_uses = kept_results - kept_uses
    if missing_uses:
        # 只在缺配对时向前补找；找到相关调用就移动起点，并从缺失集合中扣掉已经补齐的 ID。
        for index in range(adjusted - 1, -1, -1):
            uses = _tool_use_ids(messages[index])
            if uses & missing_uses:
                adjusted = index
                missing_uses -= uses
                # 所有孤立结果都补齐后立即停止向前扫描，不必把更早的无关历史全部保留。
                if not missing_uses:
                    break

    kept_provider_ids = {
        provider_id
        for message in messages[adjusted:]
        if str(message.get("role") or "") == "assistant"
        for provider_id in [_message_provider_id(message)]
        if provider_id
    }
    # 同一供应商 assistant 消息可能拆成多条存储行；继续向前保留相同 provider id 的片段，避免只留半条响应。
    while adjusted > 0:
        previous = messages[adjusted - 1]
        if str(previous.get("role") or "") != "assistant":
            break
        previous_provider_id = _message_provider_id(previous)
        if not previous_provider_id or previous_provider_id not in kept_provider_ids:
            break
        adjusted -= 1

    return adjusted


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


def _unique_preserve_order(items: list[str], limit: int = 12) -> list[str]:
    # 集合负责快速判断是否出现过，列表负责保留首次出现的顺序，二者不能只留一个。
    seen: set[str] = set()
    output: list[str] = []
    # 按原顺序逐项清理空白、跳过空项、过滤重复，达到条数上限后停止。
    for item in items:
        clean = " ".join(str(item or "").strip().split())
        if not clean:
            continue
        # 用小写键做不区分大小写的去重，但输出仍保留第一次见到的原文字面。
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


# 按 token 估算预算裁剪字符串，并在需要时保留截断后缀。
# 目标是保守控制提示长度，不是按字节截断，也不会替调用方做语义事实提取。
# 较长原文应先选择相关段落再调用这里；直接硬裁全文开头可能丢掉后面的关键证据。
# 完整文本仍应保存在来源记录中，后续精确阅读可再按窗口取回。
def clip_to_token_budget(text: str, max_tokens: int, suffix: str = "...") -> str:
    clean = str(text or "").strip()
    # 无可用预算或没有内容时直接返回空串，避免继续计算与拼接截断标记。
    if max_tokens <= 0 or not clean:
        return ""
    # 已经能放下就原样返回，不必为了使用压缩函数而强行裁剪。
    if estimate_tokens(clean) <= max_tokens:
        return clean
    # 先按粗略字符/token 比例选起点；中英文比例不同，所以还需要后面的估算校正。
    approx_chars = max(32, max_tokens * 4)
    clipped = clean[:approx_chars].rstrip()
    # 逐步缩短，直到“正文加截断后缀”整体能放进预算；后缀本身也会占 token。
    while clipped and estimate_tokens(clipped + suffix) > max_tokens:
        # 每轮减去一小段而非一个字符，降低重复估算次数；max 防止切片长度继续变负。
        clipped = clipped[: max(0, len(clipped) - 24)].rstrip()
    return f"{clipped}{suffix}" if clipped else ""


@dataclass(frozen=True)
class ContextBudget:
    max_input_tokens: int  # 本次预算采用的总额度
    reserved_response_tokens: int  # 给模型回答预留的额度
    system_tokens: int  # 项目等基础说明的额度
    memory_tokens: int  # 摘要和会话记忆的额度
    evidence_tokens: int  # 检索资料的额度
    recent_tokens: int  # 近期消息的额度
    tool_observation_tokens: int  # 本轮工具结果的额度

    @property
    def available_prompt_tokens(self) -> int:
        # 先扣掉回答需要的空间，再得到输入可用的额度；至少保留 512，避免返回零或负预算。
        return max(512, self.max_input_tokens - self.reserved_response_tokens)


# 先扣掉回答预留，得到可用于提示的总量，再按系统、记忆、证据、历史与工具分区分配。
# 分区可以避免一个很长的工具结果挤掉全部用户历史，或大量历史挤掉当前检索证据。
# 各区预算不表示必须填满；上游没有记忆/证据时，不应为了用完预算编造内容。
# 返回 ContextBudget，让所有装配函数使用同一套数字，而不是各自散落魔法常量。
def default_context_budget(model: str | None = None) -> ContextBudget:
    if settings.context_dynamic_model_window_enabled:
        max_input = max(settings.context_max_input_tokens, model_context_window_tokens(model))
    else:
        max_input = max(4096, settings.context_max_input_tokens)
    reserved = max(512, min(settings.context_reserved_response_tokens, max_input // 2))
    available = max_input - reserved  # 总额度先扣回答预留，再按各区比例分配
    return ContextBudget(
        max_input_tokens=max_input,
        reserved_response_tokens=reserved,
        system_tokens=max(512, int(available * settings.context_system_ratio)),
        memory_tokens=max(512, int(available * settings.context_memory_ratio)),
        evidence_tokens=max(512, int(available * settings.context_evidence_ratio)),
        recent_tokens=max(512, int(available * settings.context_recent_ratio)),
        tool_observation_tokens=max(512, int(available * settings.context_tool_ratio)),
    )


# 将历史消息投影到近期历史预算，并返回省略、截断和精确保留的统计。
# 优先保留近期语境，但必须理解“剩下的列表”可能已经不覆盖整个 transcript。
# 调用方依据这些统计决定能否直接回答“刚才/之前”的精确回忆，避免从残缺历史猜原文。
# 它操作输入列表的派生形态，不会删除 MessageStore 中持久消息。
def compress_messages(messages: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    # 工具记录先完成微压缩，再转成带明确来源的历史证据文本；这里不构造原生 API 的 ToolMessage。
    # 普通助手回答不加工具标签；回读指针和调用 ID 放在证据正文前，避免一般截短先丢掉来源。
    rendered: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role in {"tool", "tool_use", "tool_result"} and content.strip():
            meta = _message_meta(message)
            source = ", ".join(_tool_names(message)) or "tool"
            header = f"[Historical {role}: {source}; call_id={meta.get('tool_use_id') or meta.get('tool_call_id') or ''}]"
            recovery = meta.get("recovery")
            if recovery:
                header += f"\nRecovery: {dumps_compact(recovery)}"
            content = f"{header}\n{content}"
            role = "assistant"
        if role in {"user", "assistant"} and content.strip():
            rendered.append({"role": role, "content": content})
    kept_reversed: list[dict[str, str]] = []  # 倒序收集保留下来的近期消息，最后再还原正常时间顺序
    used = 0  # 已保留消息的估算总用量；初始还没有保留消息，所以用 0
    truncated_messages = 0  # 被截短的消息条数，从 0 开始，仅正文发生裁剪时增加
    exact_suffix_messages = 0  # 从最新端连续完整保留的消息数；用于判断最近历史能否支持精确回忆
    exact_suffix_open = True  # 最开始尚未遇到被裁短的消息，因此连续完整后缀仍可继续增长
    # 从最新消息往回装，预算不够时优先保住当前对话附近的语境，而不是最早的闲聊。
    for message in reversed(rendered):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        # 工具结果在上方已转换为带来源的证据文本；系统消息与空正文仍不计入。
        if role not in {"user", "assistant"} or not content.strip():
            continue
        # 给单条消息设置软上限，避免一条很长的回复占满整个近期历史区；96/900 是当前实现的上下界。
        per_message_budget = max(96, min(900, max_tokens // 3))
        clipped = clip_to_token_budget(content, per_message_budget)
        cost = estimate_tokens(role) + estimate_tokens(clipped)
        # 已经有保留项时，下一条会超预算就停止；首条允许先进入，再由后面的 used 检查收尾。
        if kept_reversed and used + cost > max_tokens:
            break
        kept_reversed.append({"role": role, "content": clipped})
        # 比较裁剪结果和去首尾空白后的原文，判断它是完整保留还是只剩片段。
        if clipped == content.strip():
            if exact_suffix_open:
                exact_suffix_messages += 1
        else:
            truncated_messages += 1
            # 一旦在倒序途中遇到裁剪，更早的完整消息也不再属于“从最新端连续完整”的范围。
            exact_suffix_open = False
        used += cost
        if used >= max_tokens:
            break
    # 模型读历史需要旧消息在前、新消息在后；先前倒序只是为了优先分配预算。
    kept = list(reversed(kept_reversed))
    # 工具证据也参与同一预算，因此省略数以转换后的候选数为基准。
    total_eligible_messages = len(rendered)
    return kept, {
        "input_messages": len(messages),
        "eligible_messages": total_eligible_messages,
        "kept_messages": len(kept),
        "omitted_messages": max(0, total_eligible_messages - len(kept)),
        "truncated_messages": truncated_messages,
        "exact_suffix_messages": exact_suffix_messages,
        "estimated_tokens": used,
        "budget_tokens": max_tokens,
    }


def _tool_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    route = str(message.get("route") or "").strip()
    if route:
        names.append(route)
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            name = str(call.get("tool_name") or "").strip()
            if name:
                names.append(name)
    return names


# 识别可以通过工具再获取的上下文类型，例如某些旧工具输出，而非用户原始要求。
# 微压缩优先缩这些内容，必要时仍可按来源重新读取，减少不可逆信息损失。
# 可再生是策略分类，不意味着随意丢弃其工具状态、来源 ID 或错误信息。
def is_regenerable_context_message(message: dict[str, Any]) -> bool:
    role = str(message.get("role") or "")
    meta = _message_meta(message)
    if role not in {"tool", "tool_result"} and meta.get("subtype") != "tool_result":
        return False
    if meta.get("regenerable") is False:
        return False
    haystack = " ".join(_tool_names(message)).lower()
    if not haystack:
        return False
    return any(hint in haystack for hint in COMPACTABLE_TOOL_HINTS)


# 低成本微压缩：保留近期消息，对较早且可再生的工具/上下文文本进行精简。
# 根据当前问题保留更相关片段，并记录压缩消息数与估计节省量。
# 它不向数据库写正式摘要边界，也不改变原始 transcript；只影响本次提示投影。
# 如果它已把实际提示压回可接受范围，上层可以暂不触发有模型成本的正式摘要。
def microcompact_messages(
    messages: list[dict[str, Any]],
    query: str,
    *,
    keep_recent: int | None = None,
    max_excerpt_tokens: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep = keep_recent if keep_recent is not None else settings.context_microcompact_keep_recent_messages  # 优先采用本次调用指定的近期保留条数；None 才使用全局配置，0 仍是有效输入
    # 先算近期保留段的起点，再向前修正工具调用/结果的配对边界。
    # 因此实际保留条数可能比 keep 多，不能为了凑条数拆开一组工具协议消息。
    keep_start = adjust_keep_start_to_preserve_api_invariants(
        messages,
        max(0, len(messages) - max(0, keep)),
    )
    output: list[dict[str, Any]] = []  # 存放处理后的消息；从空列表开始，每遍历一条再加入一条，不直接改原列表
    compacted = 0  # 本次实际执行微压缩的消息条数；尚未处理任何消息，所以从 0 开始累计
    tokens_before = 0  # 所有消息压缩前的估算 token 总量；从 0 开始，循环里逐条相加
    tokens_after = 0  # 处理后正文的累计估算用量；尚未处理消息时为 0，之后与处理前用同一方法比较
    compacted_tools: list[str] = []  # 收集被压缩消息关联的工具名；无压缩发生时保持空列表

    # 必须逐条检查：不同消息的年代、是否可重新获取、长度都不同，不能把整段历史一刀切。
    # index 用于判断它是否位于近期保留段之前，message 是本轮正在检查的单条消息。
    for index, message in enumerate(messages):
        content = str(message.get("content") or "")  # 当前这一条消息的原正文；缺失值按空串处理，而不是字符串 None
        before = estimate_tokens(content)  # 当前消息的估算用量，不是整个会话的总量，也不是供应商计费值
        # 无论这条最终压不压，都先计入原始总量；否则前后差值会漏掉原样保留的消息。
        tokens_before += before
        next_message = dict(message)  # 复制当前消息的字典，只替换副本的正文，避免改到调用方持有的原消息
        # 三个条件同时成立才压缩：足够旧、属于可再生工具结果、而且正文长到值得压。
        # 近期消息保留细节；短结果再包装可能反而变长；用户要求也不能当成可重跑工具内容。
        if (
            index < keep_start
            and is_regenerable_context_message(message)
            and before > max_excerpt_tokens * 2
        ):
            # 压缩不只是删字：先保留来源工具，再按当前 query 取相关片段，方便后续知道该重跑哪个工具。
            names = _tool_names(message)  # 本条消息关联的工具名，可能不止一个
            excerpt = extract_relevant_snippet(query, content, max_tokens=max_excerpt_tokens)  # 围绕当前问题选出的短摘录
            # 来源标签最多展示四个不重复工具名；没有名称时用通用标签，不猜测一个具体工具。
            tool_label = ", ".join(_unique_preserve_order(names, 4)) or "tool output"
            compacted_content = f"{MICROCOMPACT_MARKER}\nSource: {tool_label}\nRecoverable: rerun the corresponding workspace, RAG, or GitHub tool if exact details are needed."
            # 摘录不等于原始记录；保留可实际调用的 session/message 回读参数。
            recovery = _message_meta(message).get("recovery")
            if recovery:
                compacted_content += f"\nRecovery: {dumps_compact(recovery)}"
            # 只有确实提取到片段时才追加正文摘录；来源标识和恢复提示即使没有摘录也保留。
            if excerpt:
                compacted_content += f"\nKey excerpt: {excerpt}"
            # 仅替换副本正文；角色、工具调用身份和其他消息元数据仍沿用原值。
            next_message["content"] = compacted_content
            compacted += 1  # 只有真正进入压缩分支才计数；保留原样的消息不能算压缩成功
            compacted_tools.extend(names)
        # 对实际将返回的正文重新估算，包含新增来源提示的成本；不能拿摘录长度冒充最终长度。
        tokens_after += estimate_tokens(str(next_message.get("content") or ""))
        # 压缩和不压缩的消息都加入输出，保持原来的先后顺序与一进一出的条数关系。
        output.append(next_message)

    # 同时返回处理后的消息和统计，调用方既能继续组装提示，也能展示本轮节省情况。
    # 节省量用 max(0, ...) 防止极短内容加上提示标签后出现负的“节省”；该值不是实际账单。
    return output, {
        "mode": "microcompact_regenerable_context",
        "input_messages": len(messages),  # 本次检查的全部消息数，不只是工具结果条数
        "compacted_messages": compacted,  # 进入微压缩分支的条数，原样保留项不计入
        "tokens_before": tokens_before,  # 循环累计的原正文估算总量
        "tokens_after": tokens_after,  # 循环累计的实际输出正文估算总量
        "estimated_tokens_saved": max(0, tokens_before - tokens_after),
        "kept_recent_messages": max(0, min(keep, len(messages))),  # 配置层面的近期保留数；协议边界修正后实际保留范围可能更大
        "compacted_tools": _unique_preserve_order(compacted_tools, 12),  # 最多列十二个不同工具名，用于诊断，不是完整调用轨迹
    }


# 从长文本中选择与 query 更相关的局部片段，优先保留命中附近而不是永远截取开头。
# max_tokens 限制返回片段的估算规模；用于证据和日志时可减少大量无关前缀。
# 这是词面相关性提取，不是事实推理或完整摘要；未命中的上下文可能仍需工具补读。
def extract_relevant_snippet(query: str, text: str, max_tokens: int = 220) -> str:
    clean = " ".join(str(text or "").replace("\r", "\n").split())
    if not clean:
        return ""
    # 查询词去重后用于相关性打分；同一词在问题中重复出现，不应凭重复次数抬高权重。
    terms = set(_words(query))
    if not terms:
        return clip_to_token_budget(clean, max_tokens)

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(clean) if part.strip()]
    if not sentences:
        sentences = [clean]

    # 保存每句的相关分、原位置和正文；原位置用于选中后恢复阅读顺序。
    scored: list[tuple[float, int, str]] = []
    # 逐句与查询比较，避免全文只从开头截取而漏掉后半段真正相关的证据。
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        matched = [term for term in terms if term and term in lower]
        # 不同命中词数提供覆盖度，词出现次数补充局部相关性；3.0 是启发式权重，不是概率。
        score = len(set(matched)) * 3.0 + sum(lower.count(term) for term in matched)
        if query.lower().strip() and query.lower().strip() in lower:
            # 完整查询也出现在句子里时额外加分；仍只用于挑片段，不代表事实被验证。
            score += 4.0
        scored.append((score, index, sentence))

    positive = [item for item in scored if item[0] > 0]
    # 没有词面命中时退回预算裁剪，不凭空制造一段“相关证据”。
    if not positive:
        return clip_to_token_budget(clean, max_tokens)

    # 负号使高分在前；同分按原文位置排序，保证选择过程稳定。
    positive.sort(key=lambda item: (-item[0], item[1]))
    # 先取最相关的最多四句，再按原位置拼回，兼顾相关性和阅读连贯性。
    selected = sorted(positive[:4], key=lambda item: item[1])
    snippet = " ".join(sentence for _score, _index, sentence in selected)
    return clip_to_token_budget(snippet, max_tokens)


# 对候选证据做提示级压缩，同时保留来源 ID、标题、分数和检索轨迹。
# 先按 query 取相关片段，再遵守单条预算与可选总预算，避免 Top-K 条数少却每条极长。
# 候选原文和展示 snippet 分开；上游重排应先看到必要原文，再执行这里的缩短。
# 结果少了/短了要反映在统计中，不能把压缩丢失的信息仍当成模型已经看过的证据。
def compress_retrieval_results(
    query: str,
    items: list[dict[str, Any]],
    max_tokens_per_item: int = 260,
    max_total_tokens: int | None = None,
) -> list[dict[str, Any]]:
    compressed: list[dict[str, Any]] = []  # 按输入顺序累积最终可放进提示的证据副本
    total_budget = max(0, int(max_total_tokens)) if max_total_tokens is not None else None  # None 表示不设总上限，0 表示没有总预算；不能把这两种情况混在一起
    used_tokens = 0  # 当前证据集合已消耗的估算 token；没有加入任何片段前从 0 计数
    # 逐条为证据分配预算，上一条已经花掉的用量会影响下一条；不是让每条都重复使用全部总额。
    for item in items:
        # 总预算已经用尽就不再追加证据，避免 Top-K 很小但总正文仍然超长。
        if total_budget is not None and used_tokens >= total_budget:
            break
        if (
            total_budget is not None
            and compressed
            and total_budget - used_tokens < min(24, max(1, int(max_tokens_per_item)))
        ):
            break
        item_budget = max(1, int(max_tokens_per_item))
        if total_budget is not None:
            # 单条同时受自己的上限和剩余总量约束，取较小值。
            item_budget = min(item_budget, total_budget - used_tokens)
        # 优先从召回保留的原文提取；没有原文才使用展示片段，不能假设 snippet 就是完整文件。
        raw_text = str(item.get("_raw_content") or item.get("content") or item.get("snippet") or "")
        snippet = extract_relevant_snippet(query, raw_text, max_tokens=item_budget)
        original_tokens = estimate_tokens(raw_text)
        compressed_tokens = estimate_tokens(snippet)
        # 复制来源、分数等字段，但去掉大段原文，避免压过片段后又把完整 content 一并送给模型。
        next_item = {key: value for key, value in item.items() if key not in {"_raw_content", "content"}}
        next_item["snippet"] = snippet
        next_item["compression"] = {
            "mode": "query_aware_extractive",
            "original_estimated_tokens": original_tokens,
            "compressed_estimated_tokens": compressed_tokens,
            "ratio": round(compressed_tokens / original_tokens, 4) if original_tokens else 1.0,
            "item_budget_tokens": item_budget,
            "global_budget_tokens": total_budget,
            "global_tokens_before_item": used_tokens,
        }
        retrieval = dict(next_item.get("retrieval") or {})
        retrieval["context_compression"] = next_item["compression"]
        retrieval["selected_for_context"] = True
        next_item["retrieval"] = retrieval
        compressed.append(next_item)
        used_tokens += compressed_tokens
    return compressed


COMPACT_SUMMARY_SYSTEM_PROMPT = """请在上下文压缩后，为 DevFlow AI 创建一份可延续工作的摘要。
目标不是单纯减少 token，而是为软件工程 Agent 保留工作连续性。

请用纯文本返回以下部分：
1. 用户意图和明确要求。
2. 当前工作面：现在正在处理什么；如相关，请包含仓库、文件、函数、命令、测试和 UI 状态。
3. 持久决策和约束。
4. 重要事实、代码引用、错误和修复。
5. 待办任务和下一步，并以最近一次用户请求为依据。

当文件路径、标识符、日期、命令、错误消息和用户偏好很重要时，请保留原文。
不要编造细节。如果某个细节只存在于可以重新生成的工具输出中，请说明它可以通过相关工具恢复。"""

PATH_LIKE_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/])?(?:[\w.@()+\- ]+[\\/])+[\w.@()+\- ]+\.[A-Za-z0-9_]+(?::\d+)?"
)
COMMAND_HINT_PATTERN = re.compile(
    r"\b(pytest|npm|pnpm|yarn|git|docker|uvicorn|alembic|python|pip|ruff|mypy|pytest)\b",
    flags=re.IGNORECASE,
)
PRESERVE_LINE_HINTS = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "decision",
    "decide",
    "agreed",
    "todo",
    "next step",
    "prefer",
    "preference",
    "错误",
    "失败",
    "异常",
    "决定",
    "确定",
    "待办",
    "下一步",
    "偏好",
)


def _must_preserve_candidates(messages: list[dict[str, Any]], limit: int = 18) -> list[str]:
    candidates: list[str] = []
    # 逐条从待摘要原文里收集路径、命令与约束线索，作为摘要后可核对的保真候选。
    for message in messages:
        content = str(message.get("content") or "")
        if not content:
            continue
        candidates.extend(match.group(0).strip() for match in PATH_LIKE_PATTERN.finditer(content))
        # 按行检查命令/约束提示词，避免把整篇长消息作为一个必须原样保留的大字段。
        for raw_line in re.split(r"[\r\n]+", content):
            line = " ".join(raw_line.strip(" -\t").split())
            if not line:
                continue
            lower = line.lower()
            if COMMAND_HINT_PATTERN.search(line) or any(hint in lower for hint in PRESERVE_LINE_HINTS):
                candidates.append(clip_to_token_budget(line, 120))
    return _unique_preserve_order(candidates, limit)


def compaction_preservation_report(messages: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    required = _must_preserve_candidates(messages)
    summary_lower = str(summary or "").lower()
    missing = [item for item in required if item.lower() not in summary_lower]
    return {
        "required_count": len(required),
        "missing_count": len(missing),
        "required": required,
        "missing": missing,
    }


# 把原始待摘要消息里的必须保留候选，与摘要文本做对照，找出重要信息缺口。
# 必要时把遗漏项补入预算内，并给出保留报告，便于检查摘要是否丢了任务约束。
# 检查聚焦事实、任务、偏好和定位线索，不是证明摘要逐句语义等价。
# 因此精确引用、数字细节等仍可通过 transcript_ref 回原文核验。
def ensure_compaction_summary_preserves_key_facts(
    messages: list[dict[str, Any]],
    summary: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    report = compaction_preservation_report(messages, summary)
    missing = [str(item) for item in report.get("missing") or []]
    # 没有发现遗漏就只裁剪现有摘要；这只表示词面候选齐全，不是语义完整性的证明。
    if not missing:
        return clip_to_token_budget(summary, max_tokens), report

    appendix = "保真补充：\n" + "\n".join(
        f"- {clip_to_token_budget(item, 96)}" for item in missing[:12]
    )
    # 先为保真补充和分隔符预留额度，再限制摘要正文；最后仍要整体裁剪校正。
    body_budget = max(160, max_tokens - estimate_tokens(appendix) - 12)
    combined = "\n\n".join([clip_to_token_budget(summary, body_budget), appendix]).strip()
    combined = clip_to_token_budget(combined, max_tokens)
    # 组合后再查一次，裁剪也可能重新挤掉信息，不能沿用裁剪前的“已补全”判断。
    final_report = compaction_preservation_report(messages, combined)
    final_report["appended_missing_count"] = len(missing)
    return combined, final_report


def build_compaction_summary_payload(messages: list[dict[str, Any]], max_tokens: int = 6000) -> str:
    lines: list[str] = []
    # 逐条保留角色和必要正文，给摘要模型一段可追踪的时间线，而不是把消息字典整体塞入提示。
    for message in messages:
        role = str(message.get("role") or "unknown")
        if role == "system" and is_compact_boundary_message(message):
            role = "compact_boundary"
        # 先按消息数分摊单条额度；分母至少为 1 防止空输入除零，最终返回前还会检查总量。
        content = clip_to_token_budget(str(message.get("content") or ""), max(160, max_tokens // max(len(messages), 1)))
        if not content:
            continue
        tool_names = _tool_names(message)
        suffix = f" tools={tool_names}" if tool_names else ""
        lines.append(f"{role}{suffix}: {content}")
    return clip_to_token_budget("\n\n".join(lines), max_tokens)


def format_compaction_summary(summary: str) -> str:
    formatted = str(summary or "").strip()
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted, flags=re.IGNORECASE).strip()
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted, flags=re.IGNORECASE)
    if match:
        formatted = "摘要：\n" + match.group(1).strip()
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


# 模型摘要不可用时，用确定性规则组织旧消息中的任务与事实，保留基本可继续工作的背景。
# 回退结果要标记模式，诊断才能区分真实模型总结与本地规则摘要。
# 它帮助系统降级运行，不承诺保留每一句话；用户要逐字回忆仍必须读取持久原文。
def fallback_compaction_summary(messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
    user_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    assistant_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "assistant"]
    decisions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    # 模型摘要失败时，遍历双方正文做规则提取；先用户后助手，不声称还原了逐轮时间线。
    for content in [*user_messages, *assistant_messages]:
        # 每一行独立检查决策词和待办词；一行可以同时属于多个类别，不用互斥分支丢掉标签。
        for line in [part.strip(" -\t") for part in content.splitlines() if part.strip()]:
            lower = line.lower()
            if any(token in lower for token in ["decision", "decide", "agreed", "\u51b3\u5b9a", "\u786e\u5b9a"]):
                decisions.append(line)
            if any(token in lower for token in ["todo", "next step", "implement", "fix", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
                tasks.append(line)
            if len(line) > 20 and len(facts) < 10:
                facts.append(line)
    summary = "\n\n".join(
        [
            "This conversation was compacted. Continue from the preserved recent messages and this summary.",
            "User intent:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in user_messages[-8:]),
            "Assistant context:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in assistant_messages[-5:]),
            "Durable decisions:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(decisions, 12)),
            "Pending tasks:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(tasks, 12)),
            "Important facts:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(facts, 12)),
        ]
    )
    return clip_to_token_budget(summary, max_tokens or settings.context_compact_summary_tokens)


def structured_memory_from_model(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "facts": _unique_preserve_order([str(item) for item in parsed.get("facts") or []], 16),
        "decisions": _unique_preserve_order([str(item) for item in parsed.get("decisions") or []], 16),
        "open_questions": _unique_preserve_order([str(item) for item in parsed.get("open_questions") or []], 12),
        "tasks": _unique_preserve_order([str(item) for item in parsed.get("tasks") or []], 16),
        "user_preferences": _unique_preserve_order([str(item) for item in parsed.get("user_preferences") or []], 12),
        "repo_context": _unique_preserve_order([str(item) for item in parsed.get("repo_context") or []], 16),
        "citations": [item for item in parsed.get("citations") or [] if isinstance(item, dict)][:16],
    }


def fallback_memory_update(user_message: str, assistant_message: str, route: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    joined = "\n".join([user_message or "", assistant_message or ""])
    lines = [line.strip(" -\t") for line in joined.splitlines() if line.strip()]
    decisions: list[str] = []
    open_questions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    preferences: list[str] = []

    # 本地兜底逐行归类问题、决定、任务、偏好和事实，结果仍是待治理的提取，不等于人工确认。
    for line in lines:
        lower = line.lower()
        if "?" in line or "\uff1f" in line or lower.startswith(("why ", "how ", "what ")):
            open_questions.append(line)
        if any(token in lower for token in ["decide", "decision", "agreed", "chosen", "\u786e\u5b9a", "\u51b3\u5b9a"]):
            decisions.append(line)
        if any(token in lower for token in ["todo", "next step", "action", "fix", "implement", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
            tasks.append(line)
        if any(token in lower for token in ["prefer", "preference", "\u559c\u6b22", "\u504f\u597d"]):
            preferences.append(line)
        if len(line) > 20 and len(facts) < 6:
            facts.append(line)

    tools = ", ".join(str(call.get("tool_name") or "未知") for call in tool_calls) or "无"
    summary = (
        f"路由：{route}\n"
        f"用户请求：{clip_to_token_budget(user_message, 90)}\n"
        f"助手回答：{clip_to_token_budget(assistant_message, 140)}\n"
        f"工具：{tools}"
    )
    return structured_memory_from_model(
        {
            "summary": summary,
            "facts": facts,
            "decisions": decisions,
            "open_questions": open_questions,
            "tasks": tasks,
            "user_preferences": preferences,
            "repo_context": [f"使用路由：{route}"] if route else [],
            "citations": [],
        }
    )


# 将新提取的结构化字段与既有记忆合并，去重并限制各部分占用。
# digest 用于保留会话级摘要，facts/tasks/preferences/repo_context 各自对应不同信息用途。
# 通过预算防止长期运行后记忆无限增长；合并不是把所有历史正文原封不动追加一遍。
# 调用方仍需管理记忆范围与批准边界，结构合并本身不能确认事实真伪。
def merge_memory(existing: dict[str, Any], update: dict[str, Any], digest: str, max_tokens: int) -> dict[str, Any]:
    existing_summary = str(existing.get("summary") or "").strip()
    update_summary = str(update.get("summary") or digest or "").strip()
    summary_parts = [part for part in [existing_summary, update_summary] if part]
    summary = "\n\n".join(summary_parts)
    summary = clip_to_token_budget(summary, max_tokens)
    return {
        "summary": summary,
        "facts": _unique_preserve_order([*(existing.get("facts") or []), *(update.get("facts") or [])], 16),
        "decisions": _unique_preserve_order([*(existing.get("decisions") or []), *(update.get("decisions") or [])], 16),
        "open_questions": _unique_preserve_order([*(existing.get("open_questions") or []), *(update.get("open_questions") or [])], 12),
        "tasks": _unique_preserve_order([*(existing.get("tasks") or []), *(update.get("tasks") or [])], 16),
        "user_preferences": _unique_preserve_order([*(existing.get("user_preferences") or []), *(update.get("user_preferences") or [])], 12),
        "repo_context": _unique_preserve_order([*(existing.get("repo_context") or []), *(update.get("repo_context") or [])], 16),
        "citations": [*(existing.get("citations") or []), *(update.get("citations") or [])][-16:],
    }


def render_structured_memory(memory: Any, max_tokens: int | None = None) -> str:
    sections: list[str] = []
    summary = str(getattr(memory, "summary", "") or "").strip()
    if summary:
        sections.append("摘要：\n" + summary)
    # 逐类读取结构化记忆，只有该类有内容才生成小节，避免提示中充满空标题。
    for label, attr in [
        ("事实", "facts"),
        ("决策", "decisions"),
        ("未决问题", "open_questions"),
        ("任务", "tasks"),
        ("用户偏好", "user_preferences"),
        ("仓库上下文", "repo_context"),
    ]:
        values = getattr(memory, attr, None) or []
        if values:
            sections.append(label + ":\n" + "\n".join(f"- {item}" for item in values[:12]))
    text = "\n\n".join(sections)
    return clip_to_token_budget(text, max_tokens) if max_tokens else text


def memory_document_text(memory: Any) -> str:
    return render_structured_memory(memory, max_tokens=None)


def prompt_cache_model_kwargs() -> dict[str, Any]:
    if not settings.prompt_cache_enabled:
        return {}
    kwargs: dict[str, Any] = {}
    if settings.prompt_cache_key:
        kwargs["prompt_cache_key"] = settings.prompt_cache_key
    if settings.prompt_cache_retention:
        kwargs["prompt_cache_retention"] = settings.prompt_cache_retention
    return kwargs


def compression_eval_summary(before_text: str, after_text: str, retrieved: int, kept: int) -> dict[str, Any]:
    before = estimate_tokens(before_text)
    after = estimate_tokens(after_text)
    return {
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compression_ratio": round(after / before, 4) if before else 1.0,
        "retrieved_items": retrieved,
        "kept_items": kept,
        "token_savings": max(0, before - after),
    }


def dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
