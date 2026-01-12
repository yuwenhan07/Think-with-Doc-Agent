# build_context

Purpose
- Select a small, high-signal evidence set from block hits.
- Produce a context pack for the answer skill.

Input
```json
{
  "search_result": {"query": "summarize this paper", "block_hits": [...]},
  "intent": "paper_theme",
  "max_blocks": 8,
  "max_chars_per_block": 1200
}
```

Output
```json
{
  "context": {
    "question": "summarize this paper",
    "evidence": [
      {"page": 2, "block_id": "p2:b0003", "type": "text", "text": "..."}
    ]
  }
}
```
