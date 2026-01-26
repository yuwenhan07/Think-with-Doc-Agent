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


def _normalize_queries(query: str, queries: Any) -> List[str]:
    out: List[str] = []
    if isinstance(queries, list):
        for q in queries:
            q_str = str(q).strip()
            if q_str and q_str not in out:
                out.append(q_str)
    if not out and query:
        q_str = str(query).strip()
        if q_str:
            out.append(q_str)
    return out


def _tag_hits(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for h in hits:
        item = dict(h)
        item["query"] = query
        item["queries"] = [query]
        tagged.append(item)
    return tagged


def _merge_hit_queries(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    queries = list(base.get("queries") or [])
    for q in extra.get("queries") or []:
        if q not in queries:
            queries.append(q)
    base["queries"] = queries


def _merge_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Any, Dict[str, Any]] = {}
    for h in hits:
        key = h.get("id") or (h.get("page_number"), h.get("block_id"), h.get("type"))
        if key in merged:
            if float(h.get("score", 0.0)) > float(merged[key].get("score", 0.0)):
                _merge_hit_queries(h, merged[key])
                merged[key] = h
            else:
                _merge_hit_queries(merged[key], h)
        else:
            merged[key] = h
    out = list(merged.values())
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    queries = _normalize_queries(query, args.get("queries"))
    k_pages = int(args.get("k_pages", 8))
    k_blocks = int(args.get("k_blocks", 30))
    final_topk = int(args.get("final_topk", k_blocks))
    filters = args.get("filters", {}) or {}

    force_pages = filters.get("force_pages")
    if not queries:
        return {
            "query": query,
            "queries": [],
            "candidate_pages": [],
            "summary_hits": [],
            "block_hits": [],
            "stats": {"refs_ratio": 0.0, "block_hits": 0, "queries_used": 0},
        }

    summary_hits_all: List[Dict[str, Any]] = []
    block_hits_all: List[Dict[str, Any]] = []
    candidate_pages_set = set()

    for q in queries:
        if force_pages:
            qv = ctx.search_module.embed_text(q)
            block_hits = ctx.search_module.search_blocks_in_pages(
                qv=qv,
                index_dir=ctx.index_dir,
                page_nos=force_pages,
                blocks_topk=k_blocks,
                final_topk=final_topk,
                asset_base_dir=str(ctx.asset_base_dir),
            )
            summary_hits = []
            candidate_pages_set.update(force_pages)
        else:
            result = ctx.search_module.search_text_two_stage(
                query=q,
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
            candidate_pages_set.update(result.get("candidate_pages", []) or [])

        summary_hits_all.extend(_tag_hits(summary_hits, q))
        block_hits_all.extend(_tag_hits(block_hits, q))

    summary_hits_all, block_hits_all = _apply_filters(summary_hits_all, block_hits_all, filters)
    summary_hits = _merge_hits(summary_hits_all)
    block_hits = _merge_hits(block_hits_all)

    if force_pages:
        candidate_pages = list(force_pages)
    else:
        candidate_pages = [h.get("page_number") for h in summary_hits if h.get("page_number") is not None]
        candidate_pages = list(dict.fromkeys(candidate_pages))
        if not candidate_pages and candidate_pages_set:
            candidate_pages = sorted(candidate_pages_set)

    stats = _stats(summary_hits, block_hits)
    stats["queries_used"] = len(queries)

    return {
        "query": query,
        "queries": queries,
        "candidate_pages": candidate_pages,
        "summary_hits": summary_hits,
        "block_hits": block_hits,
        "stats": stats,
    }
