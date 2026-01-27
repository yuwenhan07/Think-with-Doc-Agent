from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json

_PAGE_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
_PAGES_RANGE_RE = re.compile(r"\bpages?\s*(\d+)\s*(?:-|–|—|~|to)\s*(\d+)\b", re.IGNORECASE)
_PAGE_SHORT_RE = re.compile(r"\bp\.?\s*(\d+)\b", re.IGNORECASE)
_PAGES_SHORT_RANGE_RE = re.compile(r"\bpp?\.?\s*(\d+)\s*(?:-|–|—|~|to)\s*(\d+)\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\bfig(?:ure)?\.?\s*([0-9]+[a-z]?)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\btable\.?\s*([0-9]+[a-z]?)\b", re.IGNORECASE)


def _parse_query_locators(query: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"page": None, "page_range": None, "figure": None, "table": None}
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
    return out


def _llm_parse_page_range(query: str, llm: LLMConfig) -> Optional[List[int]]:
    if not query:
        return None
    prompt = (
        "Extract page range from the query if present.\n"
        "Return JSON only. Use either:\n"
        "{\"page_range\": [start, end]}\n"
        "or {}\n"
        f"Query: {query}\n"
    )
    client = get_llm_client(llm)
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or ""
    parsed = safe_json(content) or {}
    if not isinstance(parsed, dict):
        return None
    page_range = parsed.get("page_range")
    if (
        isinstance(page_range, list)
        and len(page_range) == 2
        and all(isinstance(x, (int, float, str)) for x in page_range)
    ):
        try:
            start = int(page_range[0])
            end = int(page_range[1])
        except (TypeError, ValueError):
            return None
        if start > end:
            start, end = end, start
        return [start, end]
    return None


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


def _entry_to_evidence(entry: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    item = dict(entry)
    if item.get("type") != "text":
        item["asset_path"] = ctx.search_module.resolve_asset_path(
            item.get("asset_path"), str(ctx.asset_base_dir)
        )
    return {
        "page": item.get("page_number"),
        "block_id": item.get("block_id"),
        "type": item.get("type"),
        "text": item.get("text"),
        "asset_path": item.get("asset_path"),
        "bbox_px": item.get("bbox_px"),
    }


def _page_image_evidence(page_no: int, locator: Dict[str, Any], ctx: ExecutionContext) -> Optional[Dict[str, Any]]:
    page_entry = locator.get("pages", {}).get(str(page_no))
    if not page_entry:
        return None
    image_path = page_entry.get("image_path")
    if not image_path:
        return None
    resolved = ctx.search_module.resolve_asset_path(image_path, str(ctx.asset_base_dir))
    return {
        "page": page_no,
        "block_id": "page_image",
        "type": "page_image",
        "text": page_entry.get("text_snippet"),
        "asset_path": resolved,
        "bbox_px": None,
    }


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    locator = ctx.locator or {}
    loc = _parse_query_locators(query)
    if not loc.get("page") and not loc.get("page_range"):
        llm_range = _llm_parse_page_range(query, llm)
        if llm_range:
            loc["page_range"] = llm_range

    page_no = loc.get("page")
    page_range = loc.get("page_range")
    figure_label = loc.get("figure")
    table_label = loc.get("table")

    entries = _collect_locator_entries(locator, page_no, figure_label, table_label)
    pages: List[int] = []
    if page_no is not None:
        pages = [page_no]
    elif page_range:
        pages = list(range(page_range[0], page_range[1] + 1))
    else:
        pages = sorted({e.get("page_number") for e in entries if e.get("page_number") is not None})

    evidence: List[Dict[str, Any]] = []
    for p in pages:
        page_ev = _page_image_evidence(p, locator, ctx)
        if page_ev:
            evidence.append(page_ev)

    for entry in entries:
        evidence.append(_entry_to_evidence(entry, ctx))

    return {
        "locator": loc,
        "context": {
            "question": query,
            "evidence": evidence,
            "source": "locator",
        },
    }
