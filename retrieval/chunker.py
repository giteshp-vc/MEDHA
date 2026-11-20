import hashlib
import re
from typing import Iterable, List

from config import settings
from retrieval.models import DocumentChunk, DocumentSection


_WORD_BOUNDARY = re.compile(r"\s+")


def chunk_sections(
    sections: Iterable[DocumentSection],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[DocumentChunk]:
    """Split parsed sections into bounded chunks while preserving source locations."""
    size = int(chunk_size if chunk_size is not None else settings.MEDHA_CHUNK_SIZE_CHARS)
    overlap_size = int(overlap if overlap is not None else settings.MEDHA_CHUNK_OVERLAP_CHARS)
    if size < 100 or overlap_size < 0 or overlap_size >= size:
        raise ValueError("Chunk size must be >= 100 and overlap must be smaller than chunk size.")

    chunks: List[DocumentChunk] = []
    for section in sections:
        text = _normalize(section.text)
        if not text:
            continue
        start = 0
        local_index = 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + size // 2:
                    end = boundary + (1 if text[boundary] == " " else 0)
            content = text[start:end].strip()
            if content:
                identity = f"{section.source_id}:{section.page_number}:{section.section_heading}:{local_index}:{content}"
                chunk_id = "chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=section.source_id,
                    source_name=section.source_name,
                    source_reference=section.source_reference,
                    source_type=section.source_type,
                    retrieval_timestamp=section.retrieval_timestamp,
                    text=content,
                    page_number=section.page_number,
                    section_heading=section.section_heading,
                    chunk_index=local_index,
                    content_type=section.content_type,
                ))
            if end >= len(text):
                break
            next_start = max(start + 1, end - overlap_size)
            start = next_start
            local_index += 1
    return chunks


def _normalize(value: str) -> str:
    return _WORD_BOUNDARY.sub(" ", value or "").strip()
