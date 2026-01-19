# rewrite

Purpose
- Convert a user query into 1–N retrieval-friendly sub-queries.
- Provide negative terms and page preferences to steer search.

Input
```json
{
  "query": "Summarize the paper",
  "max_rewrites": 3
}
```

Output
```json
{
  "rewrites": ["...", "..."],
  "negative": ["references", "bibliography"],
  "page_prior": {"prefer": ["abstract", "introduction"], "avoid": ["references"]},
  "notes": "..."
}
```
