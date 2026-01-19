from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..context import ExecutionContext, LLMConfig


_REF_RE = re.compile(r"\b(references|bibliography|reference)\b", re.IGNORECASE)

def _apply_filters(
    summary_hits: List[Dict[str, Any]],
    block_hits: List[Dict[str, Any]],
    filters: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    avoid = [s.lower() for s in filters.get("avoid_sections", [])]
    if not avoid:
        return summary_hits, block_hits

    def keep_text(item: Dict[str, Any]) -> bool:
        text = (item.get("text") or item.get("summary") or "").lower()
        return not any(a in text for a in avoid)

    summary_hits = [h for h in summary_hits if keep_text(h)]
    block_hits = [h for h in block_hits if keep_text(h)]
    return summary_hits, block_hits


def _stats(summary_hits: List[Dict[str, Any]], block_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [(h.get("text") or "") for h in block_hits]
    if not texts:
        return {"refs_ratio": 0.0, "block_hits": 0}
    ref_hits = sum(1 for t in texts if _REF_RE.search(t))
    refs_ratio = ref_hits / max(1, len(texts))
    return {"refs_ratio": refs_ratio, "block_hits": len(texts)}


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    k_pages = int(args.get("k_pages", 8))
    k_blocks = int(args.get("k_blocks", 30))
    final_topk = int(args.get("final_topk", k_blocks))
    filters = args.get("filters", {}) or {}

    force_pages = filters.get("force_pages")
    if force_pages:
        qv = ctx.search_module.embed_text(query)
        block_hits = ctx.search_module.search_blocks_in_pages(
            qv=qv,
            index_dir=ctx.index_dir,
            page_nos=force_pages,
            blocks_topk=k_blocks,
            final_topk=final_topk,
            asset_base_dir=str(ctx.asset_base_dir),
        )
        summary_hits = []
        candidate_pages = list(force_pages)
    else:
        result = ctx.search_module.search_text_two_stage(
            query=query,
            index_dir=ctx.index_dir,
            summary_index=ctx.summary_index,
            summary_meta=ctx.summary_meta,
            summary_topk=k_pages,
            blocks_topk=k_blocks,
            final_topk=final_topk,
            asset_base_dir=str(ctx.asset_base_dir),
        )
        summary_hits = result.get("summary_hits", [])
        block_hits = result.get("block_hits", [])
        candidate_pages = result.get("candidate_pages", [])

    summary_hits, block_hits = _apply_filters(summary_hits, block_hits, filters)
    stats = _stats(summary_hits, block_hits)

    return {
        "query": query,
        "candidate_pages": candidate_pages,
        "summary_hits": summary_hits,
        "block_hits": block_hits,
        "stats": stats,
    }
