"""
Utility module for extracting text and determining categories for files uploaded via LINE.
Supports PDF, Word (.docx), Excel (.xlsx), CSV, JSON, Markdown, Code, Audio, and Image files.
Uses standard library zipfile and xml.etree to avoid heavy external dependencies.
"""
import io
import os
import logging
import zipfile
import xml.etree.ElementTree as ET
from typing import Tuple

logger = logging.getLogger("line_gemini_bot")

TEXT_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".json", ".md", ".py", ".js", ".ts", ".html",
    ".css", ".xml", ".log", ".yaml", ".yml", ".sql", ".sh", ".c", ".cpp",
    ".h", ".java", ".kt", ".rs", ".go", ".php", ".r", ".env", ".ini", ".conf"
}

IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".gif": "image/gif",
}

AUDIO_EXTENSIONS = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/m4a",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
}


def get_file_category(file_name: str) -> Tuple[str, str]:
    """
    Returns (category, mime_type) based on file extension.
    Categories: 'pdf', 'docx', 'xlsx', 'text', 'image', 'audio', 'unsupported'.
    """
    _, ext = os.path.splitext(file_name.lower())
    if ext == ".pdf":
        return "pdf", "application/pdf"
    elif ext == ".docx":
        return "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext in (".xlsx", ".xls"):
        return "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif ext in IMAGE_EXTENSIONS:
        return "image", IMAGE_EXTENSIONS[ext]
    elif ext in AUDIO_EXTENSIONS:
        return "audio", AUDIO_EXTENSIONS[ext]
    elif ext in TEXT_EXTENSIONS:
        if ext == ".json":
            return "text", "application/json"
        elif ext == ".csv":
            return "text", "text/csv"
        elif ext == ".html":
            return "text", "text/html"
        return "text", "text/plain"
    return "unsupported", "application/octet-stream"


def decode_text_file(file_bytes: bytes, max_chars: int = 40000) -> str:
    """
    Decodes plain text/code bytes into a string with UTF-8 and TIS-620 fallback.
    Truncates to max_chars to preserve prompt context limits.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("tis-620")
        except Exception:
            text = file_bytes.decode("utf-8", errors="replace")
    return text[:max_chars]


def extract_text_from_docx(file_bytes: bytes, max_chars: int = 40000) -> str:
    """
    Extracts text paragraphs from a Word (.docx) file binary
    using Python standard library zipfile and xml.etree.ElementTree.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            xml_content = zf.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for p in tree.iter():
                if p.tag.endswith("}p"):
                    texts = [node.text for node in p.iter() if node.tag.endswith("}t") and node.text]
                    if texts:
                        paragraphs.append("".join(texts))
            full_text = "\n".join(paragraphs)
            return full_text[:max_chars]
    except Exception as e:
        logger.warning("Failed to extract docx text: %s", e)
        return ""


def extract_text_from_xlsx(file_bytes: bytes, max_rows: int = 300) -> str:
    """
    Extracts spreadsheet data and cell values from an Excel (.xlsx) file binary
    using Python standard library zipfile and xml.etree.ElementTree.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            # 1. Read shared strings table
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                tree = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in tree.iter():
                    if si.tag.endswith("}si"):
                        texts = [t.text for t in si.iter() if t.tag.endswith("}t") and t.text]
                        if texts:
                            shared_strings.append("".join(texts))

            # 2. Read first sheet
            sheet_rows = []
            sheet_names = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
            for name in sorted(sheet_names)[:3]:  # Check up to first 3 sheets
                stree = ET.fromstring(zf.read(name))
                for row in stree.iter():
                    if row.tag.endswith("}row"):
                        cell_vals = []
                        for c in row.iter():
                            if c.tag.endswith("}c"):
                                cell_type = c.get("t")
                                v_elem = next((v for v in c.iter() if v.tag.endswith("}v")), None)
                                if v_elem is not None and v_elem.text:
                                    val = v_elem.text
                                    if cell_type == "s" and val.isdigit() and int(val) < len(shared_strings):
                                        val = shared_strings[int(val)]
                                    cell_vals.append(val)
                        if cell_vals:
                            sheet_rows.append(" | ".join(cell_vals))
                        if len(sheet_rows) >= max_rows:
                            break
                if len(sheet_rows) >= max_rows:
                    break

            if sheet_rows:
                return "\n".join(sheet_rows)
            elif shared_strings:
                return "\n".join(shared_strings[:max_rows])
    except Exception as e:
        logger.warning("Failed to extract xlsx text: %s", e)
    return ""
