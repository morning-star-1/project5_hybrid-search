import argparse
from typing import Set

from search.index import build_from_json
from search.bm25 import bm25_search, bm25_explain_terms
from search.personalize import build_user_profile_from_tags
from search.rerank import rerank


def parse_tags(tag_str: str) -> Set[str]:
    """
    Parse tags from a comma-separated string, e.g. "ranking,nlp,search"
    """
    if not tag_str:
        return set()
    return {t.strip().lower() for t in tag_str.split(",") if t.strip()}


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search Demo: BM25 retrieval + personalized re-ranking")
    parser.add_argument("--data", type=str, default="data/docs.json", help="Path to docs.json")
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to show")
    parser.add_argument("--alpha", type=float, default=0.7, help="Blend weight: final = alpha*bm25 + (1-alpha)*personal")
    parser.add_argument(
        "--user_tags",
        type=str,
        default="ranking,nlp,recommendation",
        help='Comma-separated user interest tags, e.g. "ranking,nlp,embeddings"',
    )
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b")

    args = parser.parse_args()

    # Build index
    idx = build_from_json(args.data)

    # Build user profile
    tags = parse_tags(args.user_tags)
    profile = build_user_profile_from_tags(tags)

    print("\n==============================")
    print("Hybrid Search Demo")
    print("==============================")
    print(f'Docs: {idx.N} | Unique tokens: {len(idx.postings)} | Avg doc len: {idx.avgdl:.2f}')
    print(f'Query: "{args.query}"')
    print(f"User tags: {sorted(profile.tags)}")
    print(f"alpha={args.alpha:.2f} (bm25 weight), k1={args.k1:.2f}, b={args.b:.2f}")

    matched_terms = bm25_explain_terms(idx, args.query)
    print(f"Matched query terms in index: {matched_terms}\n")

    # Stage 1: BM25
    print("=== Stage 1: BM25 (candidate retrieval) ===")
    base = bm25_search(idx, args.query, top_k=args.top_k, k1=args.k1, b=args.b)
    if not base:
        print("No results found. (Try adding more docs or changing the query.)")
        return

    for rank, (doc_id, bm25_score) in enumerate(base, start=1):
        doc = idx.docs_by_id[doc_id]
        print(f"{rank:>2}. {doc.get('title','(no title)')} [{doc_id}]")
        print(f"    bm25={bm25_score:.4f} | tags={doc.get('tags', [])}")

    # Stage 2: Re-rank with personalization
    print("\n=== Stage 2: Personalized Re-rank (final = α*bm25 + (1-α)*personal) ===")
    ranked = rerank(idx, args.query, profile, top_k=args.top_k, alpha=args.alpha)

    for rank, r in enumerate(ranked, start=1):
        print(f"{rank:>2}. {r['title']} [{r['doc_id']}]")
        print(
            f"    final={r['final']:.4f} | bm25={r['bm25']:.4f} | personal={r['personal']:.2f} | matched_tags={r['matched_tags']}"
        )

    print("\nTip for a dramatic demo: try --alpha 0.5 or pick user_tags that match one doc strongly.\n")


if __name__ == "__main__":
    main()
