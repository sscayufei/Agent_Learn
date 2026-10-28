# 把上传字节转换成可切片文本：按扩展名读取 TXT/Markdown/PDF/DOCX 等格式，并约束体积与文本长度。
# PDF 提取的是可提取文本，不等于具备扫描件 OCR；DOCX 尽量保留标题、段落和表格的结构线索。
from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path


MAX_DOCUMENT_BYTES = 12 * 1024 * 1024
MAX_EXTRACTED_CHARS = 4_000_000
MAX_PDF_PAGES = 500
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


def _ensure_text_limit(text: str) -> str:
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ValueError(f"extracted text exceeds the {MAX_EXTRACTED_CHARS} character limit")
    return text


# 将上传字节按支持的编码尝试解码，返回后续解析使用的 Unicode 文本。
# 编码解码发生在文件内容层，不要把上传文件名的编码与正文编码混为一谈。
# 无法合理解码的输入需要显式处理，不能让二进制内容带着大量乱码进入切片与 Embedding。
def decode_text(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise ValueError("binary content cannot be indexed as text")
    # 按约定顺序逐种解码，前一种失败再试下一种；latin-1 是宽松回退，不代表自动准确识别了编码。
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return _ensure_text_limit(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return ""


# 使用 PDF 文本提取器按页读取，并把页边界转换为后续可识别的文本标记。
# 页号有助于切片和引用定位；纯文本提取不能自动识别所有多栏阅读顺序或扫描图片文字。
# 因此 PDF 打开正常不等于一定能提取出可索引文本，空文本会在处理流水线中报错。
def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES} page limit")
    pages: list[str] = []  # 仅收集有可提取正文的页面，仍保留原 PDF 页号
    extracted_chars = 0  # 已经提取并准备拼接的字符总量；开始未读取页面，因此从 0 累加
    # 逐页提取才能保留引用页号，并在过程中检查累计文本上限，不等整个大文档读完才发现超限。
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            content = f"<!-- page: {page_number} -->\n{text}"
            # 除页正文和页标记外，还要计入页与页之间的两个换行，统计实际将返回的文本大小。
            extracted_chars += len(content) + (2 if pages else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"PDF text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            pages.append(content)
    return _ensure_text_limit("\n\n".join(pages))


def _paragraph_as_markdown(paragraph) -> str:
    text = (paragraph.text or "").strip()
    if not text:
        return ""
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
    heading = re.match(r"heading\s+(\d+)", style_name, re.IGNORECASE)
    if heading:
        level = min(6, max(1, int(heading.group(1))))
        return f"{'#' * level} {text}"
    lowered = style_name.lower()
    if "list bullet" in lowered:
        return f"- {text}"
    if "list number" in lowered:
        return f"1. {text}"
    return text


def _table_as_markdown(table) -> str:
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    # 将不足列补为空单元格，保证转成 Markdown 表格时每行列数一致，而不是错位对齐到其他字段。
    normalized = [row + [""] * (width - len(row)) for row in rows]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    return "\n".join([render(normalized[0]), render(["---"] * width), *(render(row) for row in normalized[1:])])


# 沿 Word 正文元素顺序提取段落和表格，而不只读取 paragraphs 列表。
# 标题转换为 Markdown 层级、表格转换为文本表格，让后面的结构切片仍能利用章节和列含义。
# 提取结果用于检索，不负责还原 Word 的分页、字体或所有版式效果。
def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise ValueError(f"DOCX exceeds the {MAX_DOCX_ENTRIES} entry limit")
            # DOCX 本质是 ZIP，压缩包很小也可能展开极大；用内部条目声明大小做一道展开体积限制。
            uncompressed_size = sum(info.file_size for info in infos)
            if uncompressed_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"DOCX expanded size exceeds the {MAX_DOCX_UNCOMPRESSED_BYTES} byte limit"
                )
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("encrypted DOCX files are not supported")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid DOCX archive") from exc

    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(BytesIO(data))
    blocks: list[str] = []  # 按原 Word 正文顺序收集段落/表格转换后的文本块
    extracted_chars = 0  # 尚未输出任何正文块，累计提取字符数先设为 0，再按块加上分隔符成本
    # 直接按正文子元素遍历，才能保持段落和表格交错的阅读顺序；分开遍历两个集合会打乱原文。
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            content = _paragraph_as_markdown(Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            content = _table_as_markdown(Table(child, document))
        else:
            content = ""
        # 空段或不支持的元素不生成占位块，只有实际提取出的文本才累计长度并追加。
        if content:
            extracted_chars += len(content) + (2 if blocks else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"DOCX text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            blocks.append(content)
    return _ensure_text_limit("\n\n".join(blocks))


# 统一文件解析入口：依据 filename 后缀选择 PDF、DOCX 或文本解码路径。
# 返回纯文本/带结构标记的文本，不返回向量或数据库模型，职责停留在“拿到可读内容”。
# 体积与文本长度由本模块限制，后续仍会检查是否为空、能否切成有效片段。
def extract_document_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    return decode_text(data)
