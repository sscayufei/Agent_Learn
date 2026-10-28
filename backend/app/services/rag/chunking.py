# 结构感知切片：先识别文档块，再在大小/重叠限制内组装 chunk，而不是固定字符数硬切全文。
# Markdown 尽量保留章节和代码围栏；CSV 保留表头；JSON/PDF 等也携带可定位元数据。
# parent_id/child_index 记录父子关系，section/行号/页号用于回溯；长块仍需降级拆分以守住大小上限。
# 文件后半部分的源码切片保留兼容用途，不代表当前 source_policy 会把所有源码放入 RAG。
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".pdf", ".docx"}
STRUCTURED_DATA_SUFFIXES = {".json", ".csv"}


# 内部结构块，先描述文档本身的段落/代码/表格，再由打包器组合成对外 chunk。
# section_path 是可读标题链；section_instance 还带出现位置，用来区分同文中重名章节。
# start_line/end_line/page 帮助回原文定位，metadata 保存格式相关补充信息。
# 使用 field(default_factory=dict) 让每个块拥有自己的元数据容器，避免多个实例共享同一个可变字典。
@dataclass
class _Block:
    text: str  # 这一个结构块的正文，不是整篇文档
    start_line: int | None  # 来源起始行；无法定位时为 None
    end_line: int | None  # 来源结束行，便于回原文补读
    kind: str = "paragraph"  # 段落、代码围栏等块类型，影响后续拆分
    section_path: tuple[str, ...] = ()  # 可读的逐级标题链
    section_instance: tuple[tuple[str, int], ...] = ()  # 标题及出现位置，区分重复同名章节
    page: int | None = None  # 来源页号，普通文本可以没有
    metadata: dict[str, Any] = field(default_factory=dict)  # 格式相关补充信息；每个实例单独创建字典


MAX_CHUNKS_PER_DOCUMENT = 5000


# 大小必须为正，重叠必须在 [0,max_chars) 内，保证切分可以持续向前推进。
# 如果 overlap 不小于块大小，相邻片段可能反复携带同一内容，无法有效减少剩余输入。
# 这里的计量是字符数，不是模型 token，也不是磁盘 UTF-8 字节数。
def _validate_limits(max_chars: int, overlap: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between zero and max_chars")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# 把多个身份片段用分隔符组合后做 SHA-256，取固定长度作为稳定内部标识。
# 显式分隔避免 (ab,c) 与 (a,bc) 简单拼接后碰成同一个字符串。
# 内容和父子位置决定 chunk 身份，便于相同输入重建时获得可重复的关联键；不是权限凭证。
def _stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _line_for_offset(text: str, offset: int, first_line: int | None) -> int | None:
    if first_line is None:
        return None
    return first_line + text.count("\n", 0, max(0, offset))


# 单个结构块仍超长时，先按句末/换行分成语义小单元，再在大小限制内重新组合。
# 没有可用边界或单句仍超限时，才回退字符窗口，确保长度约束不会被特殊文本绕过。
# 每个新块继承原章节/页号等元数据，并尽量根据文本偏移恢复行号。
# 这一步牺牲部分语义完整性换取硬上限，调用方仍需保留原文以便精确补读。
def _split_long_text(block: _Block, max_chars: int) -> list[_Block]:
    if len(block.text) <= max_chars:
        return [block]

    units = [
        unit
        for unit in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+|\n+", block.text)
        if unit and unit.strip()
    ]
    if len(units) <= 1:
        units = [block.text[index : index + max_chars] for index in range(0, len(block.text), max_chars)]

    pieces: list[str] = []
    current = ""  # 当前尚未输出的文本片；开始没有累积句子，所以为空串
    # 逐个尝试装入句子/文本单元，尽量保住自然边界；单个单元超限时才用固定字符窗口兜底。
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(unit[index : index + max_chars] for index in range(0, len(unit), max_chars))
            continue
        # 先试拼再检查长度，连接换行也计入字符数，不能只把两个正文长度相加。
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)

    output: list[_Block] = []
    cursor = 0  # 回查原文位置的起点，最初从文本第 0 个字符开始，后续移到上一个片段之后
    # 为拆出的每片回查原文偏移并换算行范围，切短后仍应保留可定位信息。
    for piece in pieces:
        offset = block.text.find(piece, cursor)
        if offset < 0:
            offset = cursor
        end_offset = min(len(block.text), offset + len(piece))
        output.append(
            _Block(
                text=piece,
                start_line=_line_for_offset(block.text, offset, block.start_line),
                end_line=_line_for_offset(block.text, max(offset, end_offset - 1), block.start_line),
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        )
        # 下一片从本片之后继续查找，避免原文重复句子时总误定位到第一次出现的位置。
        cursor = end_offset
    return output


# 超长 Markdown 代码围栏不能当普通段落随意截成不配对的围栏片段。
# 先给开/闭围栏预留字符，再将代码正文按可用体积拆分，并为各片重新补围栏。
# 当前实现使用三字符围栏标记重建包装；不要声称它保留了全部原始语言标记或完整可执行语法。
# 极小预算不足以容纳包装时退为 code_fragment，再用普通长文本策略切分。
# 定位元数据保留原文件行范围，便于查完整上下文，而不是让学员把片段直接当成可运行源码。
def _split_long_code_fence(block: _Block, max_chars: int) -> list[_Block]:
    if len(block.text) <= max_chars:
        return [block]
    lines = block.text.splitlines()
    if not lines:
        return []

    if max_chars <= 8:
        plain = _Block(
            text=block.text,
            start_line=block.start_line,
            end_line=block.end_line,
            kind="code_fragment",
            section_path=block.section_path,
            section_instance=block.section_instance,
            page=block.page,
            metadata=dict(block.metadata),
        )
        return _split_long_text(plain, max_chars)

    original_opener = lines[0] if lines[0].lstrip().startswith(("```", "~~~")) else "```"
    marker = original_opener.lstrip()[:3]
    opener = marker
    has_closer = len(lines) > 1 and lines[-1].lstrip().startswith(marker)
    body = lines[1:-1] if has_closer else lines[1:]
    closing = marker
    # 两处换行和开闭围栏都占大小；先扣包装，再算正文容量，否则“正文没超限”也可能使整个 chunk 超限。
    wrapper_size = len(opener) + len(closing) + 2
    body_limit = max_chars - wrapper_size

    if not body:
        return [
            _Block(
                text=f"{opener}\n{closing}",
                start_line=block.start_line,
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        ]

    output: list[_Block] = []
    current: list[str] = []  # 当前代码片已收集的完整行；开始尚未装入任何行
    current_size = 0  # 当前代码正文片的字符用量，从 0 起逐行累计；围栏包装成本已经另外扣除
    first_body_line = (block.start_line + 1) if block.start_line is not None else None
    current_start = first_body_line  # 当前片在原文件里的起始行，输出后更新到下一段起点

    def flush(end_line: int | None) -> None:
        nonlocal current, current_size, current_start
        # 没有待输出代码行时不制造空片，尤其处理超长单行前可能刚好已经清空。
        if not current:
            return
        body_text = "\n".join(current)
        output.append(
            _Block(
                text=f"{opener}\n{body_text}\n{closing}",
                start_line=current_start,
                end_line=end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={
                    **block.metadata,
                    "fence_start_line": block.start_line,
                    "fence_end_line": block.end_line if has_closer else None,
                },
            )
        )
        current = []
        # 本片已输出，清零是开启下一片的计数，不是把全文件已处理长度抹掉。
        current_size = 0

    # 按原代码行装片，尽量不拆一行；body_index 是去掉围栏后的局部索引，index 再换算成原文行号。
    for body_index, line in enumerate(body):
        index = (first_body_line + body_index) if first_body_line is not None else None
        if len(line) > body_limit:
            flush((index - 1) if index is not None else None)
            # 单行自身就超限时必须分段，否则无论怎样换片都放不下；这些段的原文行号仍相同。
            for offset in range(0, len(line), body_limit):
                piece = line[offset : offset + body_limit]
                output.append(
                    _Block(
                        text=f"{opener}\n{piece}\n{closing}",
                        start_line=index,
                        end_line=index,
                        kind=block.kind,
                        section_path=block.section_path,
                        section_instance=block.section_instance,
                        page=block.page,
                        metadata={
                            **block.metadata,
                            "fence_start_line": block.start_line,
                            "fence_end_line": block.end_line if has_closer else None,
                        },
                    )
                )
            current_start = (index + 1) if index is not None else None
            continue
        # 非首行拼接时要多算一个换行，防止看似满足上限的代码片实际超出。
        line_size = len(line) + (1 if current else 0)
        if current and current_size + line_size > body_limit:
            flush((index - 1) if index is not None else None)
            current_start = index
        if not current:
            current_start = index
        current.append(line)
        current_size += line_size
    last_body_line = None
    if block.end_line is not None:
        last_body_line = block.end_line - (1 if has_closer else 0)
    flush(last_body_line)
    return output


def _expand_block(block: _Block, max_chars: int) -> list[_Block]:
    if block.kind == "code_fence":
        return _split_long_code_fence(block, max_chars)
    return _split_long_text(block, max_chars)


# 从上一片尾部挑选完整块作为下一片重叠，补回跨片阅读所需的局部语境。
# 优先结构块而非固定字符，减少把代码围栏/段落截成半截的情况。
# overlap 只是目标量；打包时还要为新块和标题留空间，必要时会继续丢掉较早重叠块。
# 重叠越大不是越好：会增加相似候选重复、Embedding 成本和检索后去重压力。
def _overlap_blocks(blocks: list[_Block], overlap: int) -> list[_Block]:
    if overlap <= 0 or not blocks:
        return []
    selected: list[_Block] = []
    remaining = overlap
    # 从上一片尾部向前收集邻近语境，才能让重叠贴着新片开始处，而不是重复最早的内容。
    for block in reversed(blocks):
        # 跳过空块和代码围栏；普通重叠策略不再复制代码围栏，避免代码片重复膨胀。
        if not block.text or block.kind == "code_fence":
            continue
        separator = 2 if selected else 0
        allowance = remaining - separator
        if allowance <= 0:
            break
        if len(block.text) <= allowance:
            selected.append(block)
            remaining -= len(block.text) + separator
            continue
        offset = len(block.text) - allowance
        lookback_start = max(0, offset - min(64, max(16, allowance * 2)))
        lookback = block.text[lookback_start:offset]
        sentence_boundaries = list(re.finditer(r"[。！？.!?；;]\s*", lookback))
        word_boundaries = list(re.finditer(r"\s+", lookback))
        if sentence_boundaries:
            fragment_offset = lookback_start + sentence_boundaries[-1].end()
        elif word_boundaries:
            fragment_offset = lookback_start + word_boundaries[-1].end()
        else:
            boundary_window = block.text[offset : offset + min(48, max(8, allowance // 3))]
            boundary = re.search(r"\s+|[。！？.!?；;]\s*", boundary_window)
            fragment_offset = offset + boundary.end() if boundary else offset
        selected.append(
            _Block(
                text=block.text[fragment_offset:],
                start_line=_line_for_offset(block.text, fragment_offset, block.start_line),
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={**block.metadata, "overlap_fragment": True},
            )
        )
        remaining = 0
        break
    return list(reversed(selected))


def _section_prefix(section_path: tuple[str, ...], page: int | None, limit: int | None = None) -> str:
    parts = [f"{'#' * min(index + 1, 6)} {title}" for index, title in enumerate(section_path)]
    if page is not None:
        parts.append(f"Page {page}")
    value = "\n".join(parts)
    if limit is None or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


# 将章节路径/页号组织成片段前缀，使一个被单独召回的正文仍有主题定位。
# 前缀本身也占 max_chars，必须与正文共同预算，不能先塞满正文再无条件追加标题。
# 特别长标题在这里压缩，避免空出一个“只有标题没有正文”的超限片段。
def _prefix_for_chunk(
    section_path: tuple[str, ...],
    page: int | None,
    max_chars: int,
    *,
    has_body: bool,
) -> str:
    if not section_path and page is None:
        return ""
    if not has_body:
        limit = max_chars
    elif max_chars <= 3:
        limit = 0
    else:
        limit = min(512, max_chars // 3, max_chars - 3)
    return _section_prefix(section_path, page, limit=limit) if limit > 0 else ""


def _join_block_text(blocks: list[_Block]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text).strip()


# 把已选中的同章节块拼成最终字典，并补上可用于检索、去重和回溯的元数据。
# parent_id 由来源路径、章节实例与页号确定，child_index 是同父下面的片段顺序。
# chunk_id 再结合父标识、子序号和内容计算，区分同章不同子片。
# section_title 提供简短展示，section_path 保留完整标题链，行范围合并各组成块的最小/最大值。
# 包含代码围栏的片段标记 markdown_code，帮助后续理解它的内容类型。
def _emit_chunk(
    blocks: list[_Block],
    *,
    source_path: str,
    chunk_type: str,
    child_index: int,
    max_chars: int,
) -> dict[str, Any]:
    first = blocks[0]
    body = _join_block_text(blocks)
    prefix = _prefix_for_chunk(first.section_path, first.page, max_chars, has_body=bool(body))
    content = f"{prefix}\n\n{body}" if prefix and body else (prefix or body)
    parent_id = _stable_id(
        "parent",
        source_path,
        first.section_instance or tuple((title, 0) for title in first.section_path),
        first.page,
    )
    # 单片包含代码围栏时，内容类型需要反映这一点，不能只因为文件是 Markdown 就一律当普通段落。
    effective_type = "markdown_code" if any(block.kind == "code_fence" for block in blocks) else chunk_type
    metadata: dict[str, Any] = {}
    for block in blocks:
        metadata.update(block.metadata)
    return {
        "content": content,
        "chunk_id": _stable_id("chunk", parent_id, child_index, content),
        "parent_id": parent_id,
        "child_index": child_index,
        "chunk_type": effective_type,
        "section_path": list(first.section_path),
        "section_title": first.section_path[-1] if first.section_path else None,
        "page": first.page,
        "start_line": min((block.start_line for block in blocks if block.start_line is not None), default=None),
        "end_line": max((block.end_line for block in blocks if block.end_line is not None), default=None),
        **metadata,
    }


# 结构切片的核心打包器：在章节/页边界内装块，达到大小上限时输出片段并携带适量尾部重叠。
# current_key 把同名但不同出现位置的章节区分开，跨 key 时先 flush，避免相邻章节混成一片。
# 每个 raw_block 先扣除标题前缀预算，再由 _expand_block 拆掉仍超长的单块。
# 新块加不进当前片时，先输出旧片，再从旧片尾取重叠；若重叠挤掉新内容，则持续删除较早重叠。
# child_counts 按父 key 计数，后续父上下文扩展可据此找到相邻孩子。
# 末尾 flush 输出最后一个尚未满的片段，最终断言所有 content 都满足字符上限。
def _pack_blocks(
    blocks: list[_Block],
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    chunk_type: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []  # 已经正式输出的文档片段，和仍在拼装的 current 区分开
    current: list[_Block] = []  # 当前待合并的一组结构块；达到大小或章节边界才输出
    # key 不是只看标题文字，还看章节出现位置和页号；两个同名“配置”章节不会被误认为同一父节点。
    current_key: tuple[tuple[tuple[str, int], ...], int | None] | None = None
    child_counts: dict[tuple[tuple[tuple[str, int], ...], int | None], int] = {}  # 每个章节/页父节点已有多少子片，首次出现的父节点默认从 0 编号

    def flush() -> None:
        nonlocal current
        if not current or current_key is None:
            return
        # 当前父节点尚未输出子片时编号为 0，后续每输出一次加 1；不同父节点各自计数。
        child_index = child_counts.get(current_key, 0)
        chunks.append(
            _emit_chunk(
                current,
                source_path=source_path,
                chunk_type=chunk_type,
                child_index=child_index,
                max_chars=max_chars,
            )
        )
        child_counts[current_key] = child_index + 1

    # 外层按原文顺序处理结构块，先扣标题/页号前缀，再决定正文还有多少可用容量。
    for raw_block in blocks:
        key = (raw_block.section_instance, raw_block.page)
        prefix = _prefix_for_chunk(
            raw_block.section_path,
            raw_block.page,
            max_chars,
            has_body=bool(raw_block.text),
        )
        prefix_size = len(prefix)
        body_limit = max(1, max_chars - prefix_size - (2 if prefix_size else 0))
        # 一个原块也可能超限，因此内层处理它拆出的多个小块，每个小块再尝试装进当前片。
        for block in _expand_block(raw_block, body_limit):
            key = (block.section_instance, block.page)
            # 章节/页边界优先于装满预算；哪怕还剩空间，也先输出上一章节，避免语义归属混杂。
            if current and key != current_key:
                flush()
                current = []
            current_key = key
            candidate_body = _join_block_text([*current, block])
            prefix = _prefix_for_chunk(
                block.section_path,
                block.page,
                max_chars,
                has_body=bool(candidate_body),
            )
            # 正文、前缀以及分隔空行都计入最终长度，避免“正文合格但完整片超限”。
            candidate_size = len(candidate_body) + len(prefix) + (2 if prefix else 0)
            if current and candidate_size > max_chars:
                previous = list(current)
                flush()
                # 从上一片恢复尾部语境，而不是把全部上一片复制到新片；下面会继续检查与新内容能否共存。
                current = _overlap_blocks(previous, overlap)
                while current:
                    candidate_body = _join_block_text([*current, block])
                    if len(candidate_body) + len(prefix) + (2 if prefix else 0) <= max_chars:
                        break
                    # 先移除最早的重叠块，优先保留靠近新内容的局部上下文，同时确保循环每次都缩小。
                    current.pop(0)
            current.append(block)
    # 遍历结束后还要输出最后一组未满的块，否则文件末尾内容会丢失。
    flush()
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("chunk content exceeded max_chars")
    return chunks


def _plain_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(_Block(content, start_line, max(start_line, end_line)))
        current = []

    # 逐行以空行分隔段落，line_number 从 1 起与编辑器行号一致；输出时携带段落起止行。
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            flush(line_number - 1)
            continue
        if not current:
            start_line = line_number
        current.append(line)
    # 末尾不一定有空行，主动收尾才能把最后一段加入 blocks。
    flush(len(lines))
    return blocks


def _plain_chunks(
    text: str,
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "plain")
    cursor = 0  # 下一片的起始字符偏移；文本切片使用从 0 开始的索引
    text_length = len(text)  # 原文本总长度，作为游标终点，避免每轮重复理解字符串边界
    # 只要还有未处理文本就继续切片；每轮确定 end 并推进 cursor，最后不足一整片也要输出。
    while cursor < text_length:
        end = min(text_length, cursor + max_chars)
        if end < text_length:
            # 只在窗口后半段找自然换行边界，避免窗口开头一个换行导致片段过短。
            search_start = cursor + max(1, max_chars // 2)
            candidates = [
                text.rfind("\n\n", search_start, end),
                text.rfind("\n", search_start, end),
            ]
            # 选择更靠近窗口尾部的有效换行位置，兼顾上下文完整性与片段利用率；-1 表示没找到。
            boundary = max(candidates)
            if boundary > cursor:
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1)

        raw = text[cursor:end]
        # 去掉首尾空白后，定位也要补回左侧裁掉的长度，不能把净正文误标到窗口起点。
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        content = raw[left_trim:right_trim]
        if content:
            start_offset = cursor + left_trim
            end_offset = cursor + right_trim
            child_index = len(chunks)
            chunks.append(
                {
                    "content": content,
                    "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "chunk_type": "text",
                    "section_path": [],
                    "section_title": None,
                    "page": None,
                    "start_line": _line_for_offset(text, start_offset, 1),
                    "end_line": _line_for_offset(text, max(start_offset, end_offset - 1), 1),
                }
            )
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
        if end >= text_length:
            break
        cursor = max(cursor + 1, end - overlap)
    return chunks


# 把 Markdown 扫描成标题、段落、代码围栏等块，同时维护当前标题层级栈。
# 遇到同级/更高层标题时收缩栈，下一段才会归到正确章节；重复标题额外使用实例标识区分。
# 代码围栏内部按代码内容读取，避免把代码里的 # 误判为新的 Markdown 标题。
# 段落先缓存，结构边界到来时 flush，保留来源行号与章节关系。
# 输出的是 _Block，不是最终长度受限的 chunk，后面仍需 _pack_blocks。
def _markdown_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    headings: list[tuple[int, str, int]] = []
    heading_occurrences: dict[tuple[tuple[tuple[str, int], ...], int, str], int] = {}
    current_page: int | None = None
    current: list[str] = []
    current_start = 1
    current_kind = "paragraph"
    fence_marker: str | None = None
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    page_re = re.compile(r"^<!--\s*page\s*:\s*(\d+)\s*-->$", re.IGNORECASE)

    def section_path() -> tuple[str, ...]:
        return tuple(title for _, title, _ in headings)

    def section_instance() -> tuple[tuple[str, int], ...]:
        return tuple((title, occurrence) for _, title, occurrence in headings)

    def flush(end_line: int) -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(
                _Block(
                    content,
                    current_start,
                    max(current_start, end_line),
                    kind=current_kind,
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                )
            )
        current = []

    # Markdown 需要逐行维护标题栈与代码围栏状态，同一行在围栏内外含义不同，不能只用整篇正则硬拆。
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if fence_marker is not None:
            current.append(line)
            if stripped.startswith(fence_marker):
                flush(line_number)
                fence_marker = None
                current_kind = "paragraph"
            continue

        page_match = page_re.match(stripped)
        if page_match:
            flush(line_number - 1)
            current_page = int(page_match.group(1))
            continue

        heading_match = heading_re.match(line)
        if heading_match:
            flush(line_number - 1)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            # 遇到同级/更高级标题时弹出旧的同级及子级，留下真正父标题，随后再把新标题入栈。
            while headings and headings[-1][0] >= level:
                headings.pop()
            parent_instance = section_instance()
            occurrence_key = (parent_instance, level, title)
            # 同名标题可能反复出现，用出现次数区分章节实例，避免同名区段被拼成一个父节点。
            occurrence = heading_occurrences.get(occurrence_key, 0)
            heading_occurrences[occurrence_key] = occurrence + 1
            headings.append((level, title, occurrence))
            blocks.append(
                _Block(
                    "",
                    line_number,
                    line_number,
                    kind="heading",
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                    metadata={"heading_line": line_number},
                )
            )
            continue

        if stripped.startswith(("```", "~~~")):
            flush(line_number - 1)
            current_start = line_number
            current_kind = "code_fence"
            fence_marker = stripped[:3]
            current = [line]
            continue

        if not stripped:
            flush(line_number - 1)
            continue
        if not current:
            current_start = line_number
        current.append(line)
    flush(len(lines))
    return blocks


# 先按 JSON 结构组织对象/元素，尽量让同一项的数据保持在一起，而非按任意字符断开。
# 结构化字段有助于保留键和值之间的关系，单项过长仍会交给后续长度保护。
# 解析失败的输入需要使用兼容文本路径；内容看起来像 JSON 不代表一定符合 JSON 语法。
def _json_blocks(text: str, max_chars: int) -> list[_Block]:
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _plain_blocks(text)

    formatted = json.dumps(value, ensure_ascii=False, indent=2)
    if len(formatted) <= max_chars:
        return [
            _Block(
                formatted,
                1,
                max(1, len(text.splitlines())),
                kind="json_value",
                metadata={"json_path": "$"},
            )
        ]

    blocks: list[_Block] = []
    if isinstance(value, dict):
        items = [(f"$.{key}", {key: item}) for key, item in value.items()]
    elif isinstance(value, list):
        items = [(f"$[{index}]", item) for index, item in enumerate(value)]
    else:
        items = [("$", value)]
    # 按顶层记录逐项生成文本块并保留 JSON 路径，比从任意字符处切断 JSON 更容易追踪来源。
    for json_path, item in items:
        blocks.append(
            _Block(
                json.dumps(item, ensure_ascii=False, indent=2),
                None,
                None,
                kind="json_value",
                section_path=(json_path,),
                section_instance=((json_path, 0),),
                metadata={"json_path": json_path},
            )
        )
    return blocks


# 用 CSV 解析器处理转义和带逗号字段，不按简单 split(',') 误拆列。
# 表头会重复到各片，召回单个片段时仍知道每列含义；表头同样计入字符预算。
# 按行积累，超长单行或字段必要时再拆分，同时保留可定位的行信息。
# 输出片数也受到总量保护，避免极端上传输入生成无限多小片。
def _csv_chunks(text: str, source_path: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    try:
        if len(text) > csv.field_size_limit():
            csv.field_size_limit(len(text))
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return _pack_blocks(
            _plain_blocks(text),
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            chunk_type="text",
        )
    if not rows:
        return []

    def render(row: list[str]) -> str:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow(row)
        return buffer.getvalue()

    header = render(rows[0])
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "csv")

    def append_chunk(
        content: str,
        *,
        row_start: int,
        row_end: int,
        chunk_type: str = "csv_rows",
        row_part: int | None = None,
        row_parts: int | None = None,
        precise_lines: bool = True,
    ) -> None:
        child_index = len(chunks)
        chunks.append(
            {
                "content": content,
                "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                "parent_id": parent_id,
                "child_index": child_index,
                "chunk_type": chunk_type,
                "section_path": [],
                "section_title": None,
                "page": None,
                "start_line": row_start if precise_lines else None,
                "end_line": row_end if precise_lines else None,
                "row_start": row_start,
                "row_end": row_end,
                **({"row_part": row_part, "row_parts": row_parts} if row_part is not None else {}),
            }
        )
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")

    if header and len(header) + 1 >= max_chars:
        header_parts = [header[index : index + max_chars] for index in range(0, len(header), max_chars)] or [""]
        for part_index, part in enumerate(header_parts):
            append_chunk(
                part,
                row_start=1,
                row_end=1,
                chunk_type="csv_header_part",
                row_part=part_index + 1,
                row_parts=len(header_parts),
                precise_lines="\n" not in part,
            )
        header_prefix = ""
    else:
        header_prefix = header

    if not rows[1:]:
        if not chunks:
            append_chunk(header, row_start=1, row_end=1, precise_lines="\n" not in header)
        return chunks

    prefix_size = len(header_prefix) + (1 if header_prefix else 0)
    # 每片都会附表头，因此先扣表头前缀；数据行不能占用完整 max_chars。
    row_budget = max_chars - prefix_size
    records: list[tuple[str, int, int | None, int | None, bool]] = []
    # 第一行已经当表头，数据从第 2 行起编号；每行过长时继续拆成带分段身份的记录。
    for row_number, row in enumerate(rows[1:], start=2):
        rendered = render(row)
        parts = [rendered[index : index + row_budget] for index in range(0, len(rendered), row_budget)] or [""]
        for part_index, part in enumerate(parts):
            records.append((part, row_number, part_index + 1 if len(parts) > 1 else None, len(parts) if len(parts) > 1 else None, "\n" not in rendered))

    start = 0  # 下一片从第几条已渲染记录开始；列表索引从 0 起，不是 CSV 展示行号
    # 外层每轮产出一个 CSV 片段；内层决定本片装入多少记录，再据重叠设置推进 start。
    while start < len(records):
        selected: list[tuple[str, int, int | None, int | None, bool]] = []
        index = start
        # 逐条试装记录并检查带表头的完整长度，下一条超限就留给下一片，不把两条都截断。
        while index < len(records):
            candidate_rows = [item[0] for item in selected]
            candidate_rows.append(records[index][0])
            candidate = "\n".join([header_prefix, *candidate_rows]) if header_prefix else "\n".join(candidate_rows)
            if selected and len(candidate) > max_chars:
                break
            selected.append(records[index])
            index += 1
        selected_text = [item[0] for item in selected]
        content = "\n".join([header_prefix, *selected_text]) if header_prefix else "\n".join(selected_text)
        row_start = min(item[1] for item in selected)
        row_end = max(item[1] for item in selected)
        row_part = selected[0][2] if len(selected) == 1 else None
        row_parts = selected[0][3] if len(selected) == 1 else None
        append_chunk(
            content,
            row_start=row_start,
            row_end=row_end,
            row_part=row_part,
            row_parts=row_parts,
            precise_lines=all(item[4] for item in selected),
        )
        if index >= len(records):
            break
        if overlap <= 0:
            start = index
            continue
        size = 0  # 计算尾部重叠时尚未选入记录，重叠字符累计量先设为 0
        overlap_records = 0  # 本次从上一片尾部保留的记录条数，逐条选入时累计
        # 从本片最后一条向前选重叠，保留最靠近下一片的局部上下文。
        for record in reversed(selected):
            record_size = len(record[0])
            if overlap_records and size + record_size > overlap:
                break
            size += record_size
            overlap_records += 1
            if size >= overlap:
                break
        # 无论重叠多大，下一轮起点至少前进一条，避免大重叠配置让循环永远处理同一批记录。
        start = max(start + 1, index - overlap_records)
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("CSV chunk content exceeded max_chars")
    return chunks


# 文档切片总入口：校验长度/重叠，归一化换行，再按后缀选格式策略。
# Markdown/PDF/DOCX 的提取文本走结构感知路径，JSON/CSV 用对应结构策略，其余走普通文本。
# 返回每片 content 和父子/章节/行页元数据，调用方不必猜它来自原文哪里。
# 片数与大小都受约束，便于后台估计存储与 Embedding 工作量。
# 它不写数据库、不调用模型，因此可以用固定输入做确定性的单元回归。
def chunk_document(
    text: str,
    source_path: str = "document.txt",
    max_chars: int = 1600,
    overlap: int = 160,
) -> list[dict[str, Any]]:
    """Split a document on structural boundaries and return retrieval metadata."""

    _validate_limits(max_chars, overlap)
    if not text:
        return []
    normalized = _normalize_text(text)
    if not normalized.strip():
        return []

    suffix = Path(source_path).suffix.lower()
    if suffix == ".csv":
        return _csv_chunks(normalized, source_path, max_chars, overlap)
    if suffix in MARKDOWN_SUFFIXES:
        blocks = _markdown_blocks(normalized)
        chunk_type = "markdown_section"
    elif suffix == ".json":
        blocks = _json_blocks(normalized, max_chars)
        chunk_type = "json_value"
    else:
        return _plain_chunks(
            normalized,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
        )
    chunks = _pack_blocks(
        blocks,
        source_path=source_path,
        max_chars=max_chars,
        overlap=overlap,
        chunk_type=chunk_type,
    )
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
    return chunks


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    """Backward-compatible plain-text API used by older indexing paths."""

    return [
        chunk["content"]
        for chunk in chunk_document(text, "document.txt", max_chars=max_chars, overlap=overlap)
    ]


def _line_windows(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
    overlap_lines: int = 6,
) -> list[tuple[str, int, int]]:
    output: list[tuple[str, int, int]] = []
    # 外部行号从 1 起，Python 列表从 0 起；先做转换并限制非负。
    cursor = max(0, start_line - 1)
    stop = min(len(lines), end_line)
    # 按行窗口走完当前符号区间，每轮产出一片后继续前进，不混入后一个函数的行。
    while cursor < stop:
        if len(lines[cursor]) > max_chars:
            for offset in range(0, len(lines[cursor]), max_chars):
                output.append((lines[cursor][offset : offset + max_chars], cursor + 1, cursor + 1))
            cursor += 1
            continue
        # 新窗口尚未装入行，当前片用量从 0 重新累计，不是全文件总长度。
        size = 0
        end = cursor
        # 内层尝试扩展窗口终点，连同换行一起计算，下一行超限就停止本片。
        while end < stop:
            line_size = len(lines[end]) + (1 if end > cursor else 0)
            if end > cursor and size + line_size > max_chars:
                break
            size += line_size
            end += 1
        if end > cursor:
            content = "\n".join(lines[cursor:end]).strip("\n")
            if content:
                output.append((content, cursor + 1, end))
        if end >= stop:
            break
        cursor = max(cursor + 1, end - overlap_lines)
    return output


# 优先 ast.parse 获取 Python 类/函数区域，并将装饰器起点纳入范围。
# 顶层区域和类内方法用语法结构定位，比用正则匹配 def 更能处理嵌套与多行签名。
# AST 解析不执行用户源码，不应为了找符号 import 上传模块。
# 无法解析时由调用方降级到通用/行窗口，避免一种语法异常使整份文件完全不可读。
def _python_regions(text: str, max_chars: int) -> list[tuple[int, int, str, str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    lines = text.splitlines()
    regions: list[tuple[int, int, str, str]] = []
    cursor = 1

    def node_start(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", []) or []
        positions = [getattr(node, "lineno", 1), *(getattr(item, "lineno", 1) for item in decorators)]
        return min(positions)

    def append_node(node: ast.AST, parent: str | None = None) -> None:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        name = str(getattr(node, "name", "module"))
        symbol = f"{parent}.{name}" if parent else name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if isinstance(node, ast.ClassDef) and len("\n".join(lines[start - 1 : end])) > max_chars:
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                method_cursor = start
                first_method_start = node_start(methods[0])
                if first_method_start > method_cursor:
                    regions.append((method_cursor, first_method_start - 1, symbol, "class_header"))
                # 大类按方法细分，利用下一个方法起点确定当前区间的结束行，减少把整个大类硬塞进一个片。
                for index, method in enumerate(methods):
                    method_start = node_start(method)
                    next_start = node_start(methods[index + 1]) if index + 1 < len(methods) else end + 1
                    regions.append((method_start, next_start - 1, f"{symbol}.{method.name}", "method"))
                return
        regions.append((start, end, symbol, kind))

    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    # 先按顶层 AST 符号划分函数/类区间，同时保留符号间模块文本，避免导入与全局配置凭空消失。
    for node in top_level:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        if start > cursor:
            regions.append((cursor, start - 1, "module", "module"))
        append_node(node)
        cursor = max(cursor, end + 1)
    if cursor <= len(lines):
        regions.append((cursor, len(lines), "module", "module"))
    return [region for region in regions if region[0] <= region[1]]


def _generic_symbol(line: str, suffix: str) -> tuple[str, str] | None:
    patterns: list[tuple[re.Pattern[str], str]] = []
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        patterns = [
            (
                re.compile(
                    r"^\s*(?:export\s+(?:default\s+)?)?(?:(?:async\s+)?function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
                ),
                "symbol",
            ),
            (
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
                ),
                "function",
            ),
        ]
    elif suffix == ".go":
        patterns = [(re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_]\w*)"), "symbol")]
    elif suffix == ".rs":
        patterns = [
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)"), "symbol")
        ]
    elif suffix in {".java", ".kt", ".kts", ".cs", ".cpp", ".c", ".h", ".hpp", ".php", ".rb", ".swift", ".scala"}:
        patterns = [
            (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed|data|open)\s+)*(?:class|interface|enum|struct|trait|record)\s+([A-Za-z_]\w*)"), "type")
        ]
    elif suffix == ".sql":
        patterns = [(re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TABLE|VIEW)\s+([\w.]+)", re.IGNORECASE), "sql_object")]
    elif suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        patterns = [(re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*(?:\(\))?\s*\{"), "function")]
    for pattern, kind in patterns:
        match = pattern.match(line)
        if match:
            return match.group(1), kind
    return None


def _generic_regions(text: str, suffix: str) -> list[tuple[int, int, str, str]]:
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []
    # 非 Python 语言用当前支持的声明模式逐行寻找符号起点；这是启发式分区，不等于完整语法解析器。
    for line_number, line in enumerate(lines, start=1):
        symbol = _generic_symbol(line, suffix)
        if symbol:
            starts.append((line_number, symbol[0], symbol[1]))
    if not starts:
        return [(1, len(lines), "module", "module")] if lines else []

    regions: list[tuple[int, int, str, str]] = []
    if starts[0][0] > 1:
        regions.append((1, starts[0][0] - 1, "module", "module"))
    # 把相邻两个符号起点转成区间，最后一个符号延伸到文件末尾。
    for index, (start, name, kind) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        regions.append((start, end, name, kind))
    return regions


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or Path(path).name.lower()


# 代码片段兼容入口：选择语言与符号区域，再把区域限制到行/字符窗口。
# Python 可用 AST，其他语言走轻量符号识别，未知/解析失败时采用行窗口回退。
# 每段保留路径、符号和行号供当前代码读取/图谱用途，片段不一定能够独立编译运行。
# 当前项目 RAG 来源策略明确排除 code_file；此函数存在不等于所有源码应被向量化。
def chunk_code_text(text: str, path: str, max_chars: int = 2200) -> list[dict[str, Any]]:
    """Split source on language symbols, with parser and line-window fallbacks."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []
    suffix = Path(path).suffix.lower()
    language = _language_for_path(path)

    if suffix in MARKDOWN_SUFFIXES | STRUCTURED_DATA_SUFFIXES | {".txt"}:
        document_chunks = chunk_document(normalized, path, max_chars=max_chars, overlap=max(0, min(160, max_chars // 10)))
        output: list[dict[str, Any]] = []
        for chunk in document_chunks:
            symbol = chunk.get("section_title") or "document"
            header = f"File: {path}\nSection: {symbol}\nLines: {chunk['start_line']}-{chunk['end_line']}"
            output.append({**chunk, "symbol": symbol, "symbol_kind": chunk["chunk_type"], "language": language, "header": header})
        return output

    lines = normalized.splitlines()
    regions = _python_regions(normalized, max_chars) if suffix == ".py" else _generic_regions(normalized, suffix)
    if not regions:
        regions = [(1, len(lines), "module", "module")]

    chunks: list[dict[str, Any]] = []
    symbol_occurrences: dict[str, int] = {}
    # 逐个符号区间切窗口并记录符号身份，检索命中后能定位函数/类，而不只是文件名。
    for start_line, end_line, symbol, symbol_kind in regions:
        symbol_occurrence = symbol_occurrences.get(symbol, 0)
        # 同名符号每出现一次就推进计数，防止重复定义共享同一父片身份。
        symbol_occurrences[symbol] = symbol_occurrence + 1
        parent_id = _stable_id("code-parent", path, symbol, symbol_occurrence)
        for part_index, (content, part_start, part_end) in enumerate(
            _line_windows(lines, start_line, end_line, max_chars=max_chars)
        ):
            if not content.strip():
                continue
            chunk_id = _stable_id("code", parent_id, part_index, content)
            part_suffix = f" (part {part_index + 1})" if part_index else ""
            chunks.append(
                {
                    "content": content,
                    "start_line": part_start,
                    "end_line": part_end,
                    "symbol": symbol,
                    "symbol_kind": symbol_kind,
                    "language": language,
                    "chunk_type": "code_symbol" if symbol != "module" else "code_module",
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "child_index": part_index,
                    "symbol_occurrence": symbol_occurrence,
                    "header": (
                        f"File: {path}\nLanguage: {language}\nSymbol: {symbol}{part_suffix}\n"
                        f"Lines: {part_start}-{part_end}"
                    ),
                }
            )
    return chunks


def chunk_pr_files(files: list[dict[str, Any]], max_chars: int = 1800) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    # 每个 PR 文件分别切它的补丁并附文件名，不能把不同文件的 diff 混成一段无法定位的文本。
    for file in files:
        patch = file.get("patch") or ""
        filename = str(file.get("filename") or "patch.diff")
        # 一个补丁可能很长，逐片编号；这里关闭重叠，避免同一 diff 行因切片策略重复占用证据。
        for index, content in enumerate(chunk_text(patch, max_chars=max_chars, overlap=0)):
            chunks.append(
                {
                    "filename": filename,
                    "chunk_index": index,
                    "chunk_id": _stable_id("patch", filename, content),
                    "content": content,
                }
            )
    return chunks
