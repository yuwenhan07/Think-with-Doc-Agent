# finalize

Purpose
- Render the structured answer into a final payload for the caller.

Input
```json
{
  "answer": {"answer": "...", "citations": [...], "confidence": 0.72}
}
```

Output
```json
{
  "final_text": "...",
  "citations": [...],
  "confidence": 0.72
}
```
