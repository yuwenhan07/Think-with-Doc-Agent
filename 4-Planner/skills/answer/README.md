# answer

Purpose
- Produce a grounded answer using only the provided evidence.
- Return citations for key claims.

Input
```json
{
  "context": {
    "question": "summarize this paper",
    "evidence": [{"page": 2, "block_id": "p2:b0003", "text": "..."}]
  },
  "need_citations": true,
  "style": "short"
}
```

Output
```json
{
  "answer": "...",
  "citations": [{"page": 2, "block_id": "p2:b0003", "quote": "..."}],
  "confidence": 0.72
}
```
