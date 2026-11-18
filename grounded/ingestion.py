"""Bounded parsing and page-preserving, overlapping word windows."""

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int
    start_word: int
    text: str


def parse_document(filename: str, content: bytes, max_chars: int) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise DocumentError("Supported formats are PDF, Markdown, and UTF-8 text")
    if not content:
        raise DocumentError("The file is empty")
    pages: list[tuple[int, str]] = []
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise DocumentError("The file is not a valid PDF")
        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported; upload an unlocked copy")
            if len(reader.pages) > 200:
                raise DocumentError("PDFs are limited to 200 pages")
            total = 0
            for number, page in enumerate(reader.pages, 1):
                # Bound decompressed page streams before text extraction.
                stream = page.get_contents()
                if stream and len(stream.get_data()) > 5_000_000:
                    raise DocumentError("A PDF page is too complex to process")
                text = page.extract_text() or ""
                total += len(text)
                if total > max_chars:
                    raise DocumentError("Extracted text exceeds the document size limit")
                pages.append((number, text))
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("Could not read this PDF") from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("Text documents must use UTF-8 encoding") from exc
        if "\x00" in text:
            raise DocumentError("Binary content is not supported")
        if len(text) > max_chars:
            raise DocumentError("Text exceeds the document size limit")
        pages = [(1, text)]
    cleaned = [(page, re.sub(r"\s+", " ", text).strip()) for page, text in pages]
    cleaned = [(page, text) for page, text in cleaned if text]
    if not cleaned:
        raise DocumentError("No text found. Scanned PDFs require OCR before uploading")
    return cleaned


def make_chunks(filename: str, content: bytes, pages: list[tuple[int, str]],
                size: int, overlap: int) -> tuple[str, list[Chunk]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap smaller than size")
    document_id = hashlib.sha256(content).hexdigest()
    # Filenames are metadata only; never interpreted as disk paths.
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:200] or "document.txt"
    chunks = []
    for page, text in pages:
        words = text.split()
        if any(len(word) > 6000 for word in words):
            raise DocumentError("An unbroken text segment is too long; add whitespace before uploading")
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            while end > start + 1 and len(" ".join(words[start:end])) > 6000:
                end -= 1
            chunk_id = hashlib.sha256(f"{document_id}:{page}:{start}".encode()).hexdigest()[:24]
            chunks.append(Chunk(chunk_id, document_id, filename, page, start,
                                " ".join(words[start:end])))
            if end >= len(words):
                break
            start = max(start + 1, end - overlap)
    return document_id, chunks
