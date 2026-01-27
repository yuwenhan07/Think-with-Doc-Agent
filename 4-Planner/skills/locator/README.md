# locator

Purpose
- Deterministically locate pages/figures/tables from the query.
- Return image evidence (full page + figure/table crops if available).

Input
```json
{
  "query": "what does figure 2 in page 2 say?"
}
```

Output
```json
{
  "locator": {"page": 2, "page_range": null, "figure": "2", "table": null},
  "context": {
    "question": "what does figure 2 in page 2 say?",
    "evidence": [
      {"page": 2, "block_id": "page_image", "type": "page_image", "asset_path": "..."},
      {"page": 2, "block_id": "p2:b0005", "type": "figure", "asset_path": "..."}
    ],
    "source": "locator"
  }
}
```
