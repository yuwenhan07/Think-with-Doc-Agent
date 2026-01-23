from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image


# ----------------------------
# Text splitting
# ----------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*([-*]|\d+\.)\s+")
_IMAGE_MARK_RE = re.compile(r"^\s*<!--\s*Image\s*\(")
_TABLE_MARK_RE = re.compile(r"^\s*<!--\s*Table\s*\(")
_FIG_CAP_RE = re.compile(r"^\s*\*{0,2}Figure\s+\d+\.?\s*", re.IGNORECASE)
_TAB_CAP_RE = re.compile(r"^\s*\*{0,2}Table\s+\d+\.?\s*", re.IGNORECASE)


@dataclass
class TextChunk:
    kind: str  # heading|paragraph|list|marker
    text: str


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _group_paragraphs(lines: List[str]) -> List[TextChunk]:
    """Convert markdown lines into coarse chunks (no length control yet)."""
    chunks: List[TextChunk] = []
    buf: List[str] = []

    def flush(kind_hint: str = "paragraph") -> None:
        nonlocal buf
        if not buf:
            return
        para = "\n".join(buf).strip()
        buf = []
        if not para:
            return

        m = _HEADING_RE.match(para.splitlines()[0])
        if m:
            chunks.append(TextChunk("heading", para))
            return

        # Keep marker lines as standalone chunks (used to align with spans)
        first = para.splitlines()[0].strip()
        if _IMAGE_MARK_RE.match(first) or _TABLE_MARK_RE.match(first):
            chunks.append(TextChunk("marker", first))
            rest = "\n".join(para.splitlines()[1:]).strip()
            if rest:
                chunks.append(TextChunk("paragraph", rest))
            return

        # Lists (merge contiguous list lines)
        if _LIST_RE.match(first):
            chunks.append(TextChunk("list", para))
            return

        chunks.append(TextChunk(kind_hint, para))

    for line in lines:
        if line.strip() == "":
            flush()
            continue
        buf.append(line)
    flush()
    return chunks


def _split_by_length(text: str, *, max_chars: int = 1200, min_chars: int = 250) -> List[str]:
    """Split a long text into smaller pieces using sentence-ish boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # Prefer splitting on paragraph boundaries, then sentence boundaries.
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out: List[str] = []
    cur = ""

    def push() -> None:
        nonlocal cur
        if cur.strip():
            out.append(cur.strip())
        cur = ""

    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur += "\n\n" + p
        else:
            # If current is too small, try sentence split in p; else push cur.
            if len(cur) < min_chars and len(p) < max_chars:
                cur += "\n\n" + p
                push()
            else:
                push()
                cur = p

        if len(cur) > max_chars:
            # sentence-ish split
            parts = re.split(r"(?<=[\.!?。！？；;])\s+", cur)
            cur2 = ""
            for s in parts:
                if not s.strip():
                    continue
                if not cur2:
                    cur2 = s
                elif len(cur2) + 1 + len(s) <= max_chars:
                    cur2 += " " + s
                else:
                    out.append(cur2.strip())
                    cur2 = s
            if cur2.strip():
                out.append(cur2.strip())
            cur = ""

    push()

    # Merge very short tail
    if len(out) >= 2 and len(out[-1]) < 120:
        out[-2] = (out[-2] + "\n" + out[-1]).strip()
        out.pop()

    return out


def split_text_raw_to_text_blocks(
    text_raw: str,
    *,
    max_chars: int = 1200,
    min_chars: int = 250,
) -> List[Dict[str, Any]]:
    """Split OCR markdown text into text blocks suitable for embedding."""
    if not text_raw:
        return []

    text_raw = _normalize_newlines(text_raw)
    lines = text_raw.split("\n")
    coarse = _group_paragraphs(lines)

    blocks: List[Dict[str, Any]] = []
    for c in coarse:
        if c.kind == "marker":
            # Keep marker chunks for span alignment, but they are not embedding blocks.
            blocks.append({"type": "_marker", "text": c.text})
            continue

        parts = _split_by_length(c.text, max_chars=max_chars, min_chars=min_chars)
        for p in parts:
            if p.strip():
                blocks.append({"type": "text", "text": p.strip()})

    return blocks


# ----------------------------
# Span cropping
# ----------------------------


def _clamp_bbox(bbox: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1i = max(0, min(int(round(x1)), w - 1))
    y1i = max(0, min(int(round(y1)), h - 1))
    x2i = max(1, min(int(round(x2)), w))
    y2i = max(1, min(int(round(y2)), h))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


def _crop_span_assets_with_bbox(
    page_image_path: str,
    spans: List[Dict[str, Any]],
    *,
    out_dir: str,
    page_number: int,
    page_width_px: Optional[int] = None,
    page_height_px: Optional[int] = None,
    smart_in_w: Optional[int] = None,
    smart_in_h: Optional[int] = None,
    smart_out_w: Optional[int] = None,
    smart_out_h: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Helper: Crop all span regions and return mapping span_id -> dict with asset_path and bbox_px."""
    if not spans:
        return {}

    img = Image.open(page_image_path).convert("RGB")
    w, h = img.size

    # Determine working image (smart-resized) and its size.
    work_img = img
    work_w, work_h = w, h

    if smart_out_w and smart_out_h and int(smart_out_w) > 0 and int(smart_out_h) > 0:
        work_w, work_h = int(smart_out_w), int(smart_out_h)
        try:
            resample = Image.Resampling.BICUBIC  # Pillow >= 9
        except Exception:  # noqa: BLE001
            resample = Image.BICUBIC
        work_img = img.resize((work_w, work_h), resample)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, Dict[str, Any]] = {}
    for s in spans:
        span_id = s.get("span_id")
        if not span_id:
            continue

        bbox_px = s.get("bbox_px")
        bbox_rel = s.get("bbox_rel")

        # Compute bbox in the coordinate system of work_img.
        if bbox_px and len(bbox_px) == 4 and (work_w, work_h) != (w, h):
            # Treat bbox_px as 0~1000 normalized coords.
            x1, y1, x2, y2 = [float(v) for v in bbox_px]
            bx1 = x1 / 1000.0 * work_w
            by1 = y1 / 1000.0 * work_h
            bx2 = x2 / 1000.0 * work_w
            by2 = y2 / 1000.0 * work_h
            bbox = [bx1, by1, bx2, by2]
        elif bbox_rel and len(bbox_rel) == 4:
            # bbox_rel is assumed to be [0,1] relative coords.
            bbox = [
                float(bbox_rel[0]) * work_w,
                float(bbox_rel[1]) * work_h,
                float(bbox_rel[2]) * work_w,
                float(bbox_rel[3]) * work_h,
            ]
        elif bbox_px and len(bbox_px) == 4:
            # Last resort: treat bbox_px as absolute pixels in the original image.
            bbox = [float(v) for v in bbox_px]
        else:
            continue

        # Ensure correct ordering
        x1, y1, x2, y2 = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        bbox = [x1, y1, x2, y2]

        x1i, y1i, x2i, y2i = _clamp_bbox(bbox, w=work_w, h=work_h)
        crop = work_img.crop((x1i, y1i, x2i, y2i))

        safe_span = str(span_id).replace(":", "-")
        asset_fp = out_path / f"p{page_number:04d}_{safe_span}.png"
        crop.save(asset_fp)
        mapping[span_id] = {"asset_path": str(asset_fp), "bbox_px": [x1i, y1i, x2i, y2i], "work_size": [work_w, work_h]}

    return mapping


def crop_span_assets(
    page_image_path: str,
    spans: List[Dict[str, Any]],
    *,
    out_dir: str,
    page_number: int,
    page_width_px: Optional[int] = None,
    page_height_px: Optional[int] = None,
    smart_in_w: Optional[int] = None,
    smart_in_h: Optional[int] = None,
    smart_out_w: Optional[int] = None,
    smart_out_h: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Crop all span regions and return mapping span_id -> dict with asset_path and bbox_px.

    This implementation matches the user's OCR coordinate assumption:

    - `bbox_px` is treated as being in a 0~1000 normalized coordinate system.
    - We resize the rendered page image to `smart_resize.out_w/out_h` (when available)
      and crop within that resized image.

    Fallbacks:
    - If `smart_out_w/out_h` are missing, we crop on the original image size.
      In that case, `bbox_rel` (0~1) is preferred; otherwise we interpret `bbox_px`
      as absolute pixels.
    """
    return _crop_span_assets_with_bbox(
        page_image_path,
        spans,
        out_dir=out_dir,
        page_number=page_number,
        page_width_px=page_width_px,
        page_height_px=page_height_px,
        smart_in_w=smart_in_w,
        smart_in_h=smart_in_h,
        smart_out_w=smart_out_w,
        smart_out_h=smart_out_h,
    )


# ----------------------------
# Caption alignment (lightweight)
# ----------------------------


def _extract_captions_from_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a list of caption candidates in reading order."""
    caps: List[Dict[str, Any]] = []
    for i, b in enumerate(blocks):
        if b.get("type") != "text":
            continue
        t = b.get("text", "")
        first_line = t.strip().splitlines()[0] if t.strip() else ""
        if _FIG_CAP_RE.match(first_line):
            caps.append({"kind": "figure", "idx": i, "text": t.strip()})
        elif _TAB_CAP_RE.match(first_line):
            caps.append({"kind": "table", "idx": i, "text": t.strip()})
    return caps


def attach_captions_to_spans(
    text_blocks: List[Dict[str, Any]],
    spans: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Heuristically map span_id -> caption text.

    Strategy:
    - Collect caption candidates in reading order.
    - Assign them to spans of matching kind in span order.

    This is conservative; mismatches simply result in no caption.
    """
    caps = _extract_captions_from_text_blocks(text_blocks)

    fig_caps = [c["text"] for c in caps if c["kind"] == "figure"]
    tab_caps = [c["text"] for c in caps if c["kind"] == "table"]

    fig_spans = [s for s in spans if s.get("type") == "image"]
    tab_spans = [s for s in spans if s.get("type") == "table"]

    out: Dict[str, str] = {}
    for s, cap in zip(fig_spans, fig_caps):
        sid = s.get("span_id")
        if sid:
            out[sid] = cap
    for s, cap in zip(tab_spans, tab_caps):
        sid = s.get("span_id")
        if sid:
            out[sid] = cap
    return out


# ----------------------------
# Main: build blocks per page
# ----------------------------


def build_page_blocks(
    page: Dict[str, Any],
    *,
    assets_root: str,
    max_chars: int = 1200,
    min_chars: int = 250,
) -> List[Dict[str, Any]]:
    """Build blocks for a single page (text blocks + cropped span blocks)."""
    page_number = int(page.get("page_number") or 0)
    text_raw = page.get("text_raw", "") or ""
    section_type = page.get("section_type")
    page_section = page.get("page_section")
    section_relevance = page.get("section_relevance")
    page_image_path = page.get("image_path")
    spans = page.get("spans", []) or []
    smart = (((page.get("diagnostics") or {}).get("ocr") or {}).get("smart_resize") or {})
    smart_in_w = smart.get("in_w")
    smart_in_h = smart.get("in_h")
    smart_out_w = smart.get("out_w")
    smart_out_h = smart.get("out_h")

    # 1) Text blocks
    tb = split_text_raw_to_text_blocks(text_raw, max_chars=max_chars, min_chars=min_chars)

    # 2) Crop assets for spans
    span_asset_dir = str(Path(assets_root) / f"page_{page_number:04d}")
    span_assets: Dict[str, Dict[str, Any]] = {}
    if page_image_path and spans:
        span_assets = crop_span_assets(
            page_image_path,
            spans,
            out_dir=span_asset_dir,
            page_number=page_number,
            page_width_px=page.get("width_px"),
            page_height_px=page.get("height_px"),
            smart_in_w=smart_in_w,
            smart_in_h=smart_in_h,
            smart_out_w=smart_out_w,
            smart_out_h=smart_out_h,
        )

    # 3) Attach captions (optional heuristic)
    span_captions = attach_captions_to_spans(tb, spans) if spans else {}

    # 4) Assemble final blocks with stable IDs
    blocks: List[Dict[str, Any]] = []
    bidx = 0

    # Add span blocks first (often useful for retrieval), then text blocks.
    for s in spans:
        span_id = s.get("span_id")
        stype = s.get("type")
        bbox_px = s.get("bbox_px")
        if not span_id or not stype or not bbox_px:
            continue

        bidx += 1
        blocks.append(
            {
                "block_id": f"p{page_number}:b{bidx:04d}",
                "page_number": page_number,
                "type": "figure" if stype == "image" else "table",
                "span_id": span_id,
                "bbox_px": (span_assets.get(span_id) or {}).get("bbox_px") or bbox_px,
                "asset_path": (span_assets.get(span_id) or {}).get("asset_path"),
                "crop_work_size": (span_assets.get(span_id) or {}).get("work_size"),
                "text": span_captions.get(span_id),
                "section_type": section_type,
                "page_section": page_section,
                "section_relevance": section_relevance,
                "page_image_path": page_image_path,
                "source": "ocr_span",
            }
        )

    for b in tb:
        if b.get("type") != "text":
            # Skip markers from embedding blocks
            continue
        if not (b.get("text") or "").strip():
            continue
        bidx += 1
        blocks.append(
            {
                "block_id": f"p{page_number}:b{bidx:04d}",
                "page_number": page_number,
                "type": "text",
                "text": b["text"].strip(),
                "section_type": section_type,
                "page_section": page_section,
                "section_relevance": section_relevance,
                "page_image_path": page_image_path,
                "source": "ocr_md_rule",
            }
        )

    return blocks


def add_blocks_inplace(
    doc: Dict[str, Any],
    *,
    assets_root: str,
    pages_key: str = "pages",
    blocks_key: str = "blocks",
    max_chars: int = 1200,
    min_chars: int = 250,
) -> Dict[str, Any]:
    """Build blocks for all pages and write them into each page dict."""
    pages = doc.get(pages_key, []) or []
    for p in pages:
        p[blocks_key] = build_page_blocks(
            p,
            assets_root=assets_root,
            max_chars=max_chars,
            min_chars=min_chars,
        )
    return doc


# ----------------------------
# IO + Demo
# ----------------------------


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def demo_build_blocks_for_doc_json(
    input_json_path: str,
    output_json_path: str,
    *,
    assets_root: Optional[str] = None,
) -> None:
    """Test demo: build blocks + crop assets for a doc-level JSON."""
    doc = read_json(input_json_path)

    # Default assets_root: alongside the output JSON
    if assets_root is None:
        out_dir = str(Path(output_json_path).parent)
        assets_root = str(Path(out_dir) / "assets")

    add_blocks_inplace(doc, assets_root=assets_root)
    write_json(output_json_path, doc)
