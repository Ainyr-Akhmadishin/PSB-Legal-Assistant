"""Безопасная загрузка TXT/MD/DOCX и разбиение по юридическим пунктам."""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

MAX_FILE = 8_000_000
DATE = re.compile(r"(?<!\d)(\d{2})\.(\d{2})\.(\d{4})(?!\d)")
HEADING = re.compile(r"^\s*((?:Статья|СТАТЬЯ|ст\.|пункт|Пункт|п\.|Раздел|РАЗДЕЛ|Глава|ГЛАВА|§)\s*\d+(?:[.\-]\d+)*)\b", re.I)
NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+){1,4})[.)]?\s+")


def iso_date(text: str) -> str | None:
    match = DATE.search(text)
    if not match:
        return None
    from datetime import date
    try:
        return date(int(match[3]), int(match[2]), int(match[1])).isoformat()
    except ValueError:
        return None


def docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(t.text or "" for t in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def load_documents(source: Path) -> list[tuple[str, str]]:
    documents = []
    if source.is_dir():
        items = ((str(p.relative_to(source)), p.read_bytes()) for p in sorted(source.rglob("*")) if p.is_file() and p.stat().st_size <= MAX_FILE)
        documents.extend(_decode(items))
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            items = ((i.filename, archive.read(i)) for i in archive.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/") and i.file_size <= MAX_FILE)
            documents.extend(_decode(items))
    else:
        documents.extend(_decode([(source.name, source.read_bytes())]))
    if not documents:
        raise ValueError(f"Нет читаемых документов в {source}")
    return documents


def _decode(items):
    result = []
    for name, raw in items:
        suffix = Path(name).suffix.lower()
        if suffix not in (".txt", ".md", ".docx"):
            continue
        try:
            text = docx_text(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
        except (UnicodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
            continue
        if text.strip():
            result.append((name, text.strip()))
    return result


def _metadata(name: str, text: str) -> dict:
    title = Path(name).stem.replace("__", " / ").replace("_", " ")
    head = "\n".join(text.splitlines()[:12])
    revision = None
    match = re.search(r"(?:редакци[яи]|ред\.?|от|действует с|вступает в силу с)\s*(\d{2}\.\d{2}\.\d{4})", head + "\n" + title, re.I)
    if match:
        revision = iso_date(match.group(1))
    elif "ред 2023" in title.lower() or "ред 2025" in title.lower():
        revision = "2023-01-01" if "ред 2023" in title.lower() else "2025-01-01"
    return {"doc_name": title, "path": name, "revision_date": revision, "obsolete": "утратил силу" in (title + head).lower()}


def chunk_documents(documents: list[tuple[str, str]], max_chars: int = 1050) -> list[dict]:
    chunks = []
    for name, text in documents:
        meta = _metadata(name, text)
        section, paragraphs = "документ", []
        def flush():
            if not paragraphs:
                return
            body = "\n".join(paragraphs).strip()
            # Не теряем ни одного символа длинного раздела; заголовок повторяется в соседних частях.
            for start in range(0, len(body), max_chars):
                piece = body[start:start + max_chars].strip()
                if not piece:
                    continue
                ident = hashlib.sha256((name + section + str(start) + piece).encode()).hexdigest()[:20]
                chunks.append({**meta, "id": ident, "clause": section, "text": piece})
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            heading = HEADING.match(line) or NUMBERED.match(line)
            if heading:
                flush()
                paragraphs = []
                section = heading.group(1).strip(" .")
            elif sum(len(p) + 1 for p in paragraphs) + len(line) > max_chars and paragraphs:
                flush()
                paragraphs = []
            paragraphs.append(line)
        flush()
    return chunks
