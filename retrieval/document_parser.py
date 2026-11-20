import hashlib
import io
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

from config import settings
from retrieval.models import DocumentSection
from utils.helpers import utcnow_iso

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".html", ".htm", ".pdf", ".docx"}


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    source_id: str
    source_name: str
    source_reference: str
    source_type: str
    content_type: str
    sections: List[DocumentSection]


def parse_local_document(path: str | Path, allowed_root: Optional[Path] = None) -> ParsedDocument:
    source_path = Path(path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise DocumentParseError("Document source must be a regular file.")
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise DocumentParseError("Document path is outside the configured documents directory.") from exc
    extension = source_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported document type: {extension or '(no extension)'}")
    if source_path.stat().st_size > settings.MEDHA_MAX_DOCUMENT_BYTES:
        raise DocumentParseError("Document exceeds MEDHA_MAX_DOCUMENT_BYTES.")
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise DocumentParseError(f"Unable to read document ({type(exc).__name__}).") from exc
    return parse_document_bytes(
        payload,
        source_name=source_path.name,
        source_reference=f"local://{source_path.name}",
        source_type="USER_PROVIDED_ARTIFACT",
        content_type=_content_type_for_extension(extension),
        extension=extension,
        identity_material=str(source_path),
    )


def parse_document_bytes(
    payload: bytes,
    source_name: str,
    source_reference: str,
    source_type: str,
    content_type: Optional[str] = None,
    extension: Optional[str] = None,
    identity_material: Optional[str] = None,
) -> ParsedDocument:
    if not payload:
        raise DocumentParseError("Document is empty.")
    if len(payload) > settings.MEDHA_MAX_DOCUMENT_BYTES:
        raise DocumentParseError("Document exceeds MEDHA_MAX_DOCUMENT_BYTES.")
    extension = (extension or Path(source_name).suffix).lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported document type: {extension or '(unknown)'}")
    content_type = content_type or _content_type_for_extension(extension)
    source_id = "src_" + hashlib.sha256(
        ((identity_material or source_reference) + "\0").encode("utf-8") + payload
    ).hexdigest()[:20]
    timestamp = utcnow_iso()

    try:
        sections = _extract(payload, extension, source_id, source_name, source_reference, source_type, timestamp, content_type)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Unable to parse {extension} document ({type(exc).__name__}).") from exc
    if not any(section.text.strip() for section in sections):
        raise DocumentParseError("No extractable text was found in the document.")
    return ParsedDocument(source_id, source_name, source_reference, source_type, content_type, sections)


def _extract(payload, extension, source_id, name, reference, source_type, timestamp, content_type):
    common = dict(
        source_id=source_id,
        source_name=name,
        source_reference=reference,
        source_type=source_type,
        retrieval_timestamp=timestamp,
        content_type=content_type,
    )
    if extension == ".pdf":
        import pymupdf
        document = pymupdf.open(stream=payload, filetype="pdf")
        try:
            return [DocumentSection(**common, text=page.get_text("text"), page_number=index + 1) for index, page in enumerate(document)]
        finally:
            document.close()
    if extension == ".docx":
        from docx import Document
        document = Document(io.BytesIO(payload))
        sections = []
        heading = None
        buffer = []
        def flush():
            if buffer:
                sections.append(DocumentSection(**common, text="\n".join(buffer), section_heading=heading))
                buffer.clear()
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
            if style_name.startswith("Heading"):
                flush()
                heading = text
            else:
                buffer.append(text)
        for table in document.tables:
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            buffer.extend(row for row in rows if row.strip(" |"))
        flush()
        return sections
    text = payload.decode("utf-8-sig", errors="replace")
    if extension in {".html", ".htm"}:
        text = _HTMLTextExtractor().extract(text)
    return [DocumentSection(**common, text=text)]


def _content_type_for_extension(extension: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".rst": "text/x-rst",
        ".txt": "text/plain",
    }.get(extension, "application/octet-stream")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "tr"} and not self.hidden_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif tag in {"p", "li", "h1", "h2", "h3", "tr"} and not self.hidden_depth:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden_depth and data.strip():
            self.parts.append(data.strip())

    def extract(self, html: str) -> str:
        self.feed(html)
        return "\n".join(self.parts)
