from __future__ import annotations

from typing import Any, Dict, List

from ..context import ExecutionContext, LLMConfig


def _score_block(block: Dict[str, Any]) -> float:
    score = float(block.get("score", 0.0))
    btype = block.get("type")
    if btype == "text":
        score += 0.2
    else:
        # Promote non-text blocks slightly so images/tables can appear in context.
        score += 0.15
    return score


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    search_result = args.get("search_result", {})
    query = args.get("query") or search_result.get("query") or ""
    max_blocks = int(args.get("max_blocks", 8))
    max_chars = int(args.get("max_chars_per_block", 1200))

    block_hits = list(search_result.get("block_hits", []))
    block_hits.sort(key=_score_block, reverse=True)

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
