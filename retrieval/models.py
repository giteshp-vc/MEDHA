from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocumentSection:
    source_id: str
    source_name: str
    source_reference: str
    source_type: str
    retrieval_timestamp: str
    text: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    content_type: Optional[str] = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_id: str
    source_name: str
    source_reference: str
    source_type: str
    retrieval_timestamp: str
    text: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_index: int = 0
    content_type: Optional[str] = None

    def to_metadata(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_reference": self.source_reference,
            "source_type": self.source_type,
            "retrieval_timestamp": self.retrieval_timestamp,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    def to_evidence_dict(self) -> dict:
        metadata = self.chunk.to_metadata()
        return {
            **metadata,
            "evidence_id": self.chunk.chunk_id,
            "supporting_content": self.chunk.text,
            "source_identifier": self.chunk.source_reference,
            "evidence_location": (
                f"page {self.chunk.page_number}" if self.chunk.page_number is not None
                else self.chunk.section_heading
            ),
            "relevance_score": self.score,
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
        }
