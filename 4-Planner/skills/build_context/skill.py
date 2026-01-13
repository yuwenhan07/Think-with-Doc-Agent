from __future__ import annotations

from typing import Any, Dict, List

from ..context import ExecutionContext, LLMConfig


def _score_block(block: Dict[str, Any], intent: str) -> float:
    score = float(block.get("score", 0.0))
    btype = block.get("type")
    text = (block.get("text") or "").lower()
    if btype == "text":
        score += 0.2
    else:
        # Promote non-text blocks slightly so images/tables can appear in context.
        score += 0.15
    if intent in ("paper_theme", "theme", "summarize"):
        if text.startswith("#") or "abstract" in text:
            score += 0.4
        if "introduction" in text or "conclusion" in text:
            score += 0.2
    return score


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    search_result = args.get("search_result", {})
    query = args.get("query") or search_result.get("query") or ""
    intent = args.get("intent", "theme")
    max_blocks = int(args.get("max_blocks", 8))
    max_chars = int(args.get("max_chars_per_block", 1200))

    block_hits = list(search_result.get("block_hits", []))
    block_hits.sort(key=lambda b: _score_block(b, intent), reverse=True)

    evidence: List[Dict[str, Any]] = []
    for b in block_hits[:max_blocks]:
        text = b.get("text") or ""
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        evidence.append({
            "page": b.get("page_number"),
            "block_id": b.get("block_id"),
            "type": b.get("type"),
            "text": text,
            "bbox_px": b.get("bbox_px"),
            "asset_path": b.get("asset_path"),
            "span_id": b.get("span_id"),
            "score": b.get("score"),
        })

    return {
        "context": {
            "question": query,
            "evidence": evidence,
        }
    }
