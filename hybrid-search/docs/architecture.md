# Architecture

## Overview
- CLI app builds an inverted index from `data/docs.json`
- BM25 retrieves candidates
- Personalized re-ranker blends BM25 with tag overlap

## Data flow
Query -> BM25 -> candidate list -> re-rank -> results

## Key decisions
- Keep data in a small JSON file for easy demos
- Use simple tag-based personalization for clarity
