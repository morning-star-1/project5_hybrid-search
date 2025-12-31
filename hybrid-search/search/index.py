import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, DefaultDict

from .tokenizer import tokenize


@dataclass
class Index:
    postings: Dict[str, Dict[str, int]]      # token -> {doc_id: tf}
    df: Dict[str, int]                       # token -> document frequency
    doc_len: Dict[str, int]                  # doc_id -> length in tokens
    avgdl: float                             # average doc length
    N: int                                   # number of docs
    docs_by_id: Dict[str, Dict[str, Any]]    # doc_id -> raw doc fields (title/text/tags/etc)


def build_index(docs: List[Dict[str, Any]]) -> Index:
    postings: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    df: Dict[str, int] = {}
    doc_len: Dict[str, int] = {}
    docs_by_id: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        doc_id = str(doc["id"])
        title = doc.get("title", "")
        text = doc.get("text", "")
        tags = doc.get("tags", [])

        # index title + text (title helps a bit for small corpora)
        tokens = tokenize(f"{title} {text}")
        docs_by_id[doc_id] = {"id": doc_id, "title": title, "text": text, "tags": tags}

        counts = Counter(tokens)
        doc_len[doc_id] = sum(counts.values())

        # postings and df
        for tok, tf in counts.items():
            postings[tok][doc_id] = tf

    # compute df (how many docs contain token)
    for tok, doc_map in postings.items():
        df[tok] = len(doc_map)

    N = len(docs_by_id)
    avgdl = (sum(doc_len.values()) / N) if N else 0.0

    return Index(
        postings=dict(postings),
        df=df,
        doc_len=doc_len,
        avgdl=avgdl,
        N=N,
        docs_by_id=docs_by_id,
    )


def load_docs(json_path: str) -> List[Dict[str, Any]]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("docs.json must be a JSON array of documents.")
    return data


def build_from_json(json_path: str) -> Index:
    docs = load_docs(json_path)
    return build_index(docs)
