import math
from collections import defaultdict
from typing import Dict, List, Tuple

from .tokenizer import tokenize
from .index import Index


def bm25_search(
    index: Index,
    query: str,
    top_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    """
    Returns: list of (doc_id, bm25_score) sorted desc.
    """
    q_tokens = tokenize(query)
    if not q_tokens or index.N == 0:
        return []

    scores: Dict[str, float] = defaultdict(float)

    # Use unique query terms for standard BM25 (query term frequency optional)
    for t in set(q_tokens):
        postings = index.postings.get(t)
        if not postings:
            continue

        df = index.df.get(t, 0)
        # idf with +1 inside log to keep it positive-ish for common terms
        idf = math.log(1.0 + (index.N - df + 0.5) / (df + 0.5))

        for doc_id, tf in postings.items():
            dl = index.doc_len.get(doc_id, 0)
            denom = tf + k1 * (1.0 - b + b * (dl / index.avgdl if index.avgdl else 0.0))
            score = idf * (tf * (k1 + 1.0)) / (denom if denom else 1.0)
            scores[doc_id] += score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def bm25_explain_terms(index: Index, query: str) -> List[str]:
    """
    Optional helper: returns which query tokens actually existed in the index.
    Useful for printing explanations.
    """
    q_tokens = tokenize(query)
    return [t for t in dict.fromkeys(q_tokens) if t in index.postings]
