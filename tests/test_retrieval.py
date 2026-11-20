from pathlib import Path

import pytest

from config import settings
from retrieval.chunker import chunk_sections
from retrieval.document_parser import DocumentParseError, parse_document_bytes, parse_local_document
from retrieval.embedding_engine import EmbeddingUnavailableError
from retrieval.models import DocumentChunk, DocumentSection
from retrieval.reranker import reciprocal_rank_fusion
from retrieval.retriever import HybridRetriever, RetrievalError, _safe_source_reference, _validate_approved_url
from retrieval.store import HybridKnowledgeStore
from storage.workspace import ExecutionWorkspaceManager
from utils.privacy import storage_mode_context


class FakeEmbeddingEngine:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "car" in lowered or "automobile" in lowered or "vehicle" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def make_chunk(chunk_id, text, page=None):
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id="src_test",
        source_name="guide.txt",
        source_reference="/approved/guide.txt",
        source_type="USER_PROVIDED_ARTIFACT",
        retrieval_timestamp="2026-11-20T00:00:00+00:00",
        text=text,
        page_number=page,
        section_heading="Guide",
        chunk_index=0,
        content_type="text/plain",
    )


def test_parse_text_document_preserves_traceability(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Database\nPostgreSQL supports transactions.", encoding="utf-8")

    parsed = parse_local_document(path)

    assert parsed.source_name == "guide.md"
    assert parsed.source_reference == "local://guide.md"
    assert str(path) not in parsed.source_reference
    assert parsed.source_type == "USER_PROVIDED_ARTIFACT"
    assert parsed.source_id.startswith("src_")
    assert "PostgreSQL supports transactions" in parsed.sections[0].text
    assert parsed.sections[0].retrieval_timestamp


def test_parse_pdf_preserves_page_numbers():
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Page one PostgreSQL reference")
    payload = document.tobytes()
    document.close()

    parsed = parse_document_bytes(
        payload,
        source_name="reference.pdf",
        source_reference="C:/docs/reference.pdf",
        source_type="USER_PROVIDED_ARTIFACT",
        extension=".pdf",
    )

    assert parsed.sections[0].page_number == 1
    assert "PostgreSQL reference" in parsed.sections[0].text


def test_parse_docx_extracts_heading_and_paragraph():
    from docx import Document
    from io import BytesIO

    document = Document()
    document.add_heading("Database setup", level=1)
    document.add_paragraph("Configure the PostgreSQL connection string.")
    buffer = BytesIO()
    document.save(buffer)

    parsed = parse_document_bytes(
        buffer.getvalue(),
        source_name="guide.docx",
        source_reference="C:/docs/guide.docx",
        source_type="USER_PROVIDED_ARTIFACT",
        extension=".docx",
    )

    assert parsed.sections
    assert parsed.sections[0].section_heading == "Database setup"
    assert "connection string" in parsed.sections[0].text


def test_unsupported_format_is_rejected(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"not a supported document")
    with pytest.raises(DocumentParseError, match="Unsupported document type"):
        parse_local_document(path)


def test_chunking_preserves_page_and_heading_metadata():
    section = DocumentSection(
        source_id="src_1", source_name="a.txt", source_reference="a.txt",
        source_type="USER_PROVIDED_ARTIFACT", retrieval_timestamp="now",
        text="First paragraph. " * 40, page_number=3, section_heading="Database",
    )
    chunks = chunk_sections([section], chunk_size=180, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.page_number == 3 and chunk.section_heading == "Database" for chunk in chunks)
    assert all(len(chunk.chunk_id) > 8 for chunk in chunks)


def test_hybrid_store_fuses_bm25_and_semantic_results():
    store = HybridKnowledgeStore(None, FakeEmbeddingEngine(), hybrid_search=True)
    store.add_chunks([
        make_chunk("chunk_car", "A car has wheels and a motor."),
        make_chunk("chunk_automobile", "An automobile is a road vehicle."),
        make_chunk("chunk_database", "A database stores structured records."),
    ])

    results = store.retrieve("automobile", limit=3)

    assert results
    assert {item.chunk.chunk_id for item in results} >= {"chunk_automobile", "chunk_car"}
    assert any(item.semantic_rank is not None for item in results)
    assert any(item.keyword_rank is not None for item in results)
    store.close()


def test_keyword_search_tolerates_sql_and_url_punctuation():
    store = HybridKnowledgeStore(None, FakeEmbeddingEngine(), hybrid_search=False)
    store.add_chunks([make_chunk("chunk_uri", "Use postgresql psycopg2 connection URI.")])
    assert store.retrieve("postgresql+psycopg2://db:5432/app", limit=1)
    store.close()


def test_retrieval_metadata_is_ready_for_evidence_model():
    item = make_chunk("chunk_1", "PostgreSQL supports transactions.", page=7)
    retrieved = reciprocal_rank_fusion([item], [item], limit=1)
    result = retrieved[0][0]
    assert result.to_metadata()["source_id"] == "src_test"
    assert result.to_metadata()["page_number"] == 7


def test_url_fetch_requires_explicit_approved_domain(monkeypatch):
    monkeypatch.setattr(settings, "MEDHA_ALLOWED_SOURCE_DOMAINS", ())
    with pytest.raises(RetrievalError, match="not in MEDHA_ALLOWED_SOURCE_DOMAINS"):
        _validate_approved_url("https://docs.example.org/guide.html")


def test_url_fetch_rejects_non_https_and_private_addresses(monkeypatch):
    monkeypatch.setattr(settings, "MEDHA_ALLOWED_SOURCE_DOMAINS", ("example.org",))
    with pytest.raises(RetrievalError, match="HTTPS"):
        _validate_approved_url("http://docs.example.org/guide.html")
    monkeypatch.setattr("retrieval.retriever.socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 443))])
    with pytest.raises(RetrievalError, match="non-public"):
        _validate_approved_url("https://docs.example.org/guide.html")


def test_source_reference_drops_url_query_and_fragment():
    assert _safe_source_reference("https://docs.example.org/path?token=secret#section") == (
        "https://docs.example.org/path"
    )


@pytest.mark.asyncio
async def test_ephemeral_pdf_hybrid_retrieval_cites_source_and_leaves_no_files(tmp_path):
    manager = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
    )
    workspace = manager.create("synthetic-pdf-retrieval")
    uploads = workspace.subdirectory("uploads")
    pdf_path = uploads / "synthetic-guide.pdf"

    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Page one fixture material is copper.")
    document.new_page().insert_text((72, 72), "Page two source fact: the Northstar token belongs to navigation.")
    document.save(str(pdf_path))
    document.close()

    store = HybridKnowledgeStore(None, FakeEmbeddingEngine(), hybrid_search=True)
    retriever = HybridRetriever(store)
    workspace.register_closer(store.close)

    try:
        with storage_mode_context("ephemeral"):
            assert retriever.ingest_local([pdf_path]) == 2
            results = await retriever.retrieve("Northstar token navigation", limit=2)
            assert results
            evidence = results[0].to_evidence_dict()
            assert results[0].chunk.page_number == 2
            assert "Northstar token belongs to navigation" in evidence["supporting_content"]
            assert evidence["source_name"] == "synthetic-guide.pdf"
            assert evidence["source_reference"] == "local://synthetic-guide.pdf"
            assert evidence["evidence_id"] == evidence["chunk_id"]
            assert results[0].semantic_rank is not None
            assert results[0].keyword_rank is not None
            assert store.database_path is None
            assert not list(workspace.path.rglob("*.sqlite*"))

            empty_store = HybridKnowledgeStore(None, FakeEmbeddingEngine(), hybrid_search=True)
            empty_retriever = HybridRetriever(empty_store)
            workspace.register_closer(empty_store.close)
            assert await empty_retriever.retrieve("missing source query") == []
    finally:
        cleanup = workspace.close()

    assert cleanup.verified is True
    assert not pdf_path.exists()
    assert list((tmp_path / "ephemeral").iterdir()) == []
    assert list(tmp_path.rglob("*.sqlite*")) == []
