# judge_locator

Purpose
- Decide whether locator evidence accurately and completely matches the query constraints.
- Signal the planner to answer directly when locator coverage is complete.

Input
```json
{
  "query": "what does figure 2 in page 2 say?",
  "locator_result": {
    "locator": {"page": 2, "figure": "2", "table": null},
    "context": {"evidence": []}
  }
}
```

Output
```json
{
  "verdict": "good",
  "state": "complete"
}
```
