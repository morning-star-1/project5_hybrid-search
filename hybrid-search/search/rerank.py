from typing import Dict, Any, List

from .index import Index
from .bm25 import bm25_search
from .personalize import UserProfile, personalization_score, personalization_explain


def rerank(
    index: Index,
    query: str,
    profile: UserProfile,
    top_k: int = 10,
    alpha: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Stage 1: BM25 candidate retrieval
    Stage 2: Personalized re-ranking

    Returns a list of result dicts with scores + explanation fields.
    """
    candidates = bm25_search(index, query, top_k=top_k)
    results: List[Dict[str, Any]] = []

    for doc_id, bm25 in candidates:
        personal = personalization_score(index, doc_id, profile)
        final = alpha * bm25 + (1.0 - alpha) * personal

        doc = index.docs_by_id[doc_id]
        exp = personalization_explain(index, doc_id, profile)

        results.append(
            {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "bm25": bm25,
                "personal": personal,
                "final": final,
                "matched_tags": exp["matched_tags"],
                "tags": doc.get("tags", []),
            }
        )

    results.sort(key=lambda r: r["final"], reverse=True)
    return results
