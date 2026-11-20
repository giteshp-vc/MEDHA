from typing import Dict, Iterable, List

from retrieval.models import DocumentChunk


def reciprocal_rank_fusion(
    lexical: Iterable[DocumentChunk],
    semantic: Iterable[DocumentChunk],
    limit: int,
    rank_constant: int = 60,
) -> List[tuple[DocumentChunk, float, int | None, int | None]]:
    """Fuse independent BM25 and semantic rankings without mixing their score scales."""
    lexical_list = list(lexical)
    semantic_list = list(semantic)
    chunks: Dict[str, DocumentChunk] = {}
    lexical_rank: Dict[str, int] = {}
    semantic_rank: Dict[str, int] = {}
    for rank, chunk in enumerate(lexical_list, 1):
        chunks[chunk.chunk_id] = chunk
        lexical_rank[chunk.chunk_id] = rank
    for rank, chunk in enumerate(semantic_list, 1):
        chunks.setdefault(chunk.chunk_id, chunk)
        semantic_rank[chunk.chunk_id] = rank
    scores = {
        chunk_id: (1.0 / (rank_constant + lexical_rank[chunk_id]) if chunk_id in lexical_rank else 0.0)
        + (1.0 / (rank_constant + semantic_rank[chunk_id]) if chunk_id in semantic_rank else 0.0)
        for chunk_id in chunks
    }
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:limit]
    return [
        (chunks[chunk_id], scores[chunk_id], semantic_rank.get(chunk_id), lexical_rank.get(chunk_id))
        for chunk_id in ordered
    ]
