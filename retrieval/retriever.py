import asyncio
import ipaddress
import socket
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse, urlunsplit

import httpx

from config import settings
from retrieval.chunker import chunk_sections
from retrieval.document_parser import DocumentParseError, ParsedDocument, parse_document_bytes, parse_local_document
from retrieval.embedding_engine import FastEmbedEngine
from retrieval.models import DocumentChunk, RetrievedChunk
from retrieval.store import HybridKnowledgeStore


class RetrievalError(RuntimeError):
    pass


class HybridRetriever:
    """One execution's document ingestion and hybrid search service."""

    def __init__(self, store: HybridKnowledgeStore, max_chunks: Optional[int] = None):
        self.store = store
        self.max_chunks = max_chunks or settings.MEDHA_MAX_RETRIEVED_CHUNKS
        self.sources: dict[str, dict] = {}

    @classmethod
    def for_workspace(cls, workspace, max_chunks: Optional[int] = None) -> "HybridRetriever":
        database_path = None if workspace.ephemeral else workspace.path / "retrieval.sqlite3"
        cache_dir = str(workspace.path / "model-cache") if workspace.mode == "persistent" else None
        engine = FastEmbedEngine(settings.MEDHA_EMBEDDING_MODEL, cache_dir=cache_dir)
        store = HybridKnowledgeStore(
            database_path=database_path,
            embedding_engine=engine,
            hybrid_search=settings.MEDHA_HYBRID_SEARCH,
        )
        workspace.register_closer(store.close)
        return cls(store, max_chunks=max_chunks)

    def ingest_local(self, paths: Iterable[str | Path], allowed_root: Optional[Path] = None) -> int:
        sections = []
        for path in paths:
            try:
                document = parse_local_document(path, allowed_root=allowed_root)
            except DocumentParseError as exc:
                raise RetrievalError(str(exc)) from exc
            self.sources[document.source_id] = _source_record(document)
            sections.extend(document.sections)
        return self._store_sections(sections)

    async def ingest_urls(self, urls: Iterable[str]) -> int:
        sections = []
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout, limits=limits) as client:
            for initial_url in urls:
                url = _validate_approved_url(initial_url)
                response = None
                for _ in range(4):
                    status_code, headers, body = await _bounded_get(client, url)
                    if status_code not in {301, 302, 303, 307, 308}:
                        response = (status_code, headers, body)
                        break
                    location = headers.get("location")
                    if not location:
                        raise RetrievalError("Approved source redirected without a Location header.")
                    url = _validate_approved_url(str(httpx.URL(url).join(location)))
                if response is None or response[0] >= 400:
                    status = response[0] if response is not None else "unknown"
                    raise RetrievalError(f"Approved source fetch failed with HTTP status {status}.")
                content_type = response[1].get("content-type", "application/octet-stream").split(";", 1)[0].strip().lower()
                extension = _extension_for_source(url, content_type)
                source_name = Path(urlparse(url).path).name or urlparse(url).hostname or "remote-source"
                safe_reference = _safe_source_reference(url)
                try:
                    document = parse_document_bytes(
                        response[2],
                        source_name=source_name,
                        source_reference=safe_reference,
                        source_type="TOOL_RETRIEVED",
                        content_type=content_type,
                        extension=extension,
                        identity_material=url,
                    )
                except DocumentParseError as exc:
                    raise RetrievalError(str(exc)) from exc
                self.sources[document.source_id] = _source_record(document)
                sections.extend(document.sections)
        return await asyncio.to_thread(self._store_sections, sections)

    def ingest_local_async(self, paths: Iterable[str | Path], allowed_root: Optional[Path] = None) -> int:
        sections = []
        for path in paths:
            document = parse_local_document(path, allowed_root=allowed_root)
            self.sources[document.source_id] = _source_record(document)
            sections.extend(document.sections)
        return self._store_sections(sections)

    def _store_sections(self, sections) -> int:
        chunks = chunk_sections(sections)
        try:
            return self.store.add_chunks(chunks)
        except Exception as exc:
            raise RetrievalError(f"Unable to index local document chunks ({type(exc).__name__}).") from exc

    async def retrieve(self, query: str, limit: Optional[int] = None) -> List[RetrievedChunk]:
        selected = self.max_chunks if limit is None else max(0, min(limit, self.max_chunks))
        try:
            return await asyncio.to_thread(self.store.retrieve, query, selected)
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError(f"Local retrieval failed ({type(exc).__name__}).") from exc

    @property
    def source_count(self) -> int:
        return len(self.sources)


def _source_record(document: ParsedDocument) -> dict:
    return {
        "source_id": document.source_id,
        "source_name": document.source_name,
        "source_reference": document.source_reference,
        "source_type": document.source_type,
        "content_type": document.content_type,
        "retrieval_timestamp": document.sections[0].retrieval_timestamp if document.sections else None,
    }


async def _bounded_get(client: httpx.AsyncClient, url: str) -> tuple[int, httpx.Headers, bytes]:
    async with client.stream("GET", url, headers={"Accept": "text/html,application/pdf,text/plain,application/json"}) as response:
        chunks = []
        size = 0
        async for part in response.aiter_bytes():
            size += len(part)
            if size > settings.MEDHA_MAX_SOURCE_RESPONSE_BYTES:
                raise RetrievalError("Approved source exceeded MEDHA_MAX_SOURCE_RESPONSE_BYTES.")
            chunks.append(part)
        return response.status_code, response.headers, b"".join(chunks)


def _validate_approved_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RetrievalError("Online sources must be credential-free HTTPS URLs.")
    host = parsed.hostname.lower().rstrip(".")
    allowed = settings.MEDHA_ALLOWED_SOURCE_DOMAINS
    if not allowed or not any(host == domain or host.endswith("." + domain) for domain in allowed):
        raise RetrievalError("Online source domain is not in MEDHA_ALLOWED_SOURCE_DOMAINS.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise RetrievalError("Unable to resolve approved source host.") from exc
    if not addresses:
        raise RetrievalError("Approved source host resolved to no addresses.")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise RetrievalError("Approved source resolves to a non-public network address.")
    return url


def _extension_for_source(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".pdf", ".docx", ".txt", ".md", ".rst", ".csv", ".json", ".html", ".htm"}:
        return suffix
    return {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/html": ".html",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/json": ".json",
        "text/csv": ".csv",
    }.get(content_type, "")


def _safe_source_reference(url: str) -> str:
    parsed = urlparse(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
