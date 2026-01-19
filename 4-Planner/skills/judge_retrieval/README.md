# judge_retrieval

Purpose
- Decide whether retrieval is usable, references-heavy, or missing key sections.
- Suggest next actions (rewrite or search with filters).

Input
```json
{
  "query": "summarize this paper",
  "search_result": {"summary_hits": [], "block_hits": [], "stats": {"refs_ratio": 0.6}}
}
```

Output
```json
{
  "verdict": "bad",
  "reasons": ["references_heavy", "missing_abstract"],
  "suggestions": [
    {"action": "rewrite"},
    {"action": "search", "filters": {"force_pages": [1, 2, 3]}}
  ]
}
```
