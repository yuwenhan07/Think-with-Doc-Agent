# search

Purpose
- Run two-stage retrieval (summary -> block shards) using existing FAISS indexes.
- Return normalized hits and simple stats for downstream judging.

Input
```json
{
  "query": "summarize this paper",
  "queries": ["summarize this paper", "paper contributions", "main findings"],
  "page": 2,
  "k_pages": 8,
  "k_blocks": 30,
  "final_topk": 10,
  "filters": {"avoid_sections": ["references"]}
}
```

Output
```json
{
  "query": "summarize this paper",
  "queries": ["summarize this paper", "paper contributions", "main findings"],
  "candidate_pages": [2, 6, 1],
  "summary_hits": [...],
  "block_hits": [...],
  "stats": {"refs_ratio": 0.2, "block_hits": 30, "queries_used": 3}
}
```
