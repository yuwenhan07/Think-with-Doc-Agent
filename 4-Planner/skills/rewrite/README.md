# rewrite

Purpose
- Convert a user query into 1–N retrieval-friendly sub-queries.
- Provide intent, negative terms, and page preferences to steer search.

Input
```json
{
  "query": "Summarize the paper",
  "mode": "theme",
  "max_rewrites": 3
}
```

Output
```json
{
  "intent": "paper_theme",
  "rewrites": ["...", "..."],
  "negative": ["references", "bibliography"],
  "page_prior": {"prefer": ["abstract", "introduction"], "avoid": ["references"]},
  "notes": "..."
}
```
