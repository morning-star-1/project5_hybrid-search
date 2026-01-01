# Hybrid Search Demo

Hybrid Search Demo is a CLI app that blends BM25 retrieval with a simple personalization re-ranker for search relevance demos.

## Quickstart
### Prerequisites
- Python 3.11+

### Run locally
```bash
python app.py --query "ranking" --top_k 5
```

Data lives in `data/docs.json` and can be edited to customize results.

## Tests
```bash
python -m unittest discover -s tests
```
