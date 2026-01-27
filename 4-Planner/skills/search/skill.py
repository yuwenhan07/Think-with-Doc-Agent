from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..context import ExecutionContext, LLMConfig


_REF_RE = re.compile(r"\b(references|bibliography|reference)\b", re.IGNORECASE)
_PAGE_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
_PAGES_RANGE_RE = re.compile(r"\bpages?\s*(\d+)\s*(?:-|–|—|~|to)\s*(\d+)\b", re.IGNORECASE)
_PAGE_SHORT_RE = re.compile(r"\bp\.?\s*(\d+)\b", re.IGNORECASE)
_PAGES_SHORT_RANGE_RE = re.compile(r"\bpp?\.?\s*(\d+)\s*(?:-|–|—|~|to)\s*(\d+)\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\bfig(?:ure)?\.?\s*([0-9]+[a-z]?)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\btable\.?\s*([0-9]+[a-z]?)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\bsection\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)

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


def _parse_query_locators(query: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"page": None, "page_range": None, "figure": None, "table": None, "section": None}
    if not query:
        return out
    m = _PAGES_RANGE_RE.search(query)
    if not m:
        m = _PAGES_SHORT_RANGE_RE.search(query)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        if start > end:
            start, end = end, start
        out["page_range"] = [start, end]
    m = _PAGE_RE.search(query)
    if not m:
        m = _PAGE_SHORT_RE.search(query)
    if m:
        out["page"] = int(m.group(1))
    m = _FIGURE_RE.search(query)
    if m:
        out["figure"] = m.group(1)
    m = _TABLE_RE.search(query)
    if m:
        out["table"] = m.group(1)
    m = _SECTION_RE.search(query)
    if m:
        out["section"] = m.group(1)
    return out


def _collect_locator_entries(
    locator: Dict[str, Any],
    page_no: Optional[int],
    figure_label: Optional[str],
    table_label: Optional[str],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if figure_label:
        entries.extend(locator.get("figures", {}).get(str(figure_label), []))
    if table_label:
        entries.extend(locator.get("tables", {}).get(str(table_label), []))
    if page_no is None:
        return entries
    return [e for e in entries if e.get("page_number") == page_no]


def _locator_entries_to_hits(
    entries: List[Dict[str, Any]],
    ctx: ExecutionContext,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        if item.get("type") != "text":
            item["asset_path"] = ctx.search_module.resolve_asset_path(
                item.get("asset_path"), str(ctx.asset_base_dir)
            )
        hit = ctx.search_module.pretty_hit(item, 1.0)
        hit["score_reason"] = "locator"
        hits.append(hit)
    return hits


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    if "page" not in args and "page_number" in args:
        args["page"] = args.get("page_number")
    queries = _normalize_queries(query, args.get("queries"))
    k_pages = int(args.get("k_pages", 8))
    k_blocks = int(args.get("k_blocks", 30))
    final_topk = int(args.get("final_topk", k_blocks))
    filters = args.get("filters", {}) or {}

    force_pages = filters.get("force_pages")
    page_arg = args.get("page")
    if page_arg is not None and not force_pages:
        if isinstance(page_arg, list):
            force_pages = [int(p) for p in page_arg if p is not None]
        else:
            force_pages = [int(page_arg)]
    locator = ctx.locator or {}
    locator_hits: List[Dict[str, Any]] = []

    if not force_pages and query:
        loc = _parse_query_locators(query)
        page_no = loc.get("page")
        page_range = loc.get("page_range")
        figure_label = loc.get("figure")
        table_label = loc.get("table")
        section_label = loc.get("section")

        locator_hits = _locator_entries_to_hits(
            _collect_locator_entries(locator, page_no, figure_label, table_label),
            ctx,
        )

        if page_no is not None:
            force_pages = [page_no]
        elif page_range:
            force_pages = list(range(page_range[0], page_range[1] + 1))
        elif figure_label:
            entries = locator.get("figures", {}).get(str(figure_label), [])
            force_pages = sorted({e.get("page_number") for e in entries if e.get("page_number") is not None}) or None
        elif table_label:
            entries = locator.get("tables", {}).get(str(table_label), [])
            force_pages = sorted({e.get("page_number") for e in entries if e.get("page_number") is not None}) or None
        elif section_label:
            entries = locator.get("sections", {}).get(str(section_label), [])
            force_pages = sorted({e.get("page_number") for e in entries if e.get("page_number") is not None}) or None

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

    if locator_hits:
        block_hits_all.extend(_tag_hits(locator_hits, query))

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
    stats["locator_hits"] = len(locator_hits)

    return {
        "query": query,
        "queries": queries,
        "candidate_pages": candidate_pages,
        "summary_hits": summary_hits,
        "block_hits": block_hits,
        "stats": stats,
    }
