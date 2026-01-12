# judge_answer

Purpose
- Verify that the answer is supported by evidence and citations.
- Decide whether to finalize, revise, or fetch more evidence.

Input
```json
{
  "query": "summarize this paper",
  "context": {"question": "...", "evidence": [...]},
  "answer": {"answer": "...", "citations": [...]}
}
```

Output
```json
{
  "verdict": "need_more_evidence",
  "issues": ["citation_missing_for_key_claim"],
  "next_actions": [
    {"action": "search", "query_hint": "abstract introduction contributions", "filters": {"avoid_sections": ["references"]}}
  ]
}
```
