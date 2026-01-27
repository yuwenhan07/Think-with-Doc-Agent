from __future__ import annotations

import os
import time
import json
import traceback
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qwen_vl_utils import smart_resize

from openai import OpenAI
import os
import base64

# local
# BASE_URL = "http://localhost:8003/v1"
# BASE_MODEL = "Qwen3-VL-32B-Instruct"
# remote
BASE_URL = "https://qianfan.baidubce.com/v2"
BASE_MODEL = "qwen3-vl-32b-instruct"
# Base64 encoding helper
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# @title inference function with API
def inference_with_api(image_path, prompt, model_id=BASE_MODEL, min_pixels=512*32*32, max_pixels=2048*32*32):
    base64_image = encode_image(image_path)
    api_key = os.environ.get("QianFan_API_KEY")
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
       
    )
    return completion.choices[0].message.content


@dataclass
class OCRConfig:
    prompt: str = "qwenvl markdown"
    model_id: str = BASE_MODEL
    min_pixels: int = 512 * 32 * 32
    max_pixels: int = 4608 * 32 * 32
    factor: int = 32
    max_retries: int = 2
    retry_backoff_sec: float = 1.5
    sleep_between_pages_sec: float = 0.2
    cache_dir: str = "../out/ocr_cache"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ---- markdown post-processing: extract image/table bboxes + normalize to relative coords ----
_BBOX_PATTERNS = {
    "table": re.compile(r"<!--\s*Table\s*\((\s*[-\d\.]+\s*),\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\s*-->", re.IGNORECASE),
    "image": re.compile(r"<!--\s*Image\s*\((\s*[-\d\.]+\s*),\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\s*-->", re.IGNORECASE),
}


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _px_to_rel_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> List[float]:
    # Normalize to [0,1] in the order [x1, y1, x2, y2]
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    rx1 = _clamp01(x1 / float(w))
    ry1 = _clamp01(y1 / float(h))
    rx2 = _clamp01(x2 / float(w))
    ry2 = _clamp01(y2 / float(h))
    # Ensure top-left/bottom-right ordering
    if rx2 < rx1:
        rx1, rx2 = rx2, rx1
    if ry2 < ry1:
        ry1, ry2 = ry2, ry1
    return [rx1, ry1, rx2, ry2]


def strip_markdown_code_fences(text: str) -> str:
    """
    Remove leading/trailing markdown code fences such as:
      ```markdown
      ...
      ```
    or
      ```
      ...
      ```
    Only strips the outermost fence if present.
    """
    if not text:
        return text

    s = text.strip()

    # Match ``` or ```lang at start, and ``` at end
    if s.startswith("```"):
        # remove first line (``` or ```lang)
        lines = s.splitlines()
        if len(lines) >= 2:
            # drop first line
            lines = lines[1:]
            # drop last line if it's a closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

    return s

def extract_media_regions_from_markdown(markdown: str, page_w_px: int, page_h_px: int) -> List[Dict[str, Any]]:
    """Extract table/image regions from omni-parsing style markdown comments.

    Expected markers:
      <!-- Table (x1, y1, x2, y2) -->
      <!-- Image (x1, y1, x2, y2) -->

    Returns a list of regions:
      {"type": "table"|"image", "bbox_rel": [x1,y1,x2,y2], "bbox_px": [x1,y1,x2,y2]}

    Notes:
      - If no markers exist, returns [].
      - Coordinates are assumed to be in pixels on the rendered page image.
    """
    if not markdown:
        return []

    regions: List[Dict[str, Any]] = []

    for t, pat in _BBOX_PATTERNS.items():
        for m in pat.finditer(markdown):
            x1 = float(m.group(1))
            y1 = float(m.group(2))
            x2 = float(m.group(3))
            y2 = float(m.group(4))
            bbox_px = [x1, y1, x2, y2]
            bbox_rel = _px_to_rel_bbox(x1, y1, x2, y2, page_w_px, page_h_px)
            regions.append({
                "type": "table" if t == "table" else "image",
                "bbox_px": bbox_px,
                "bbox_rel": bbox_rel,
            })

    # Stable order: by top then left
    regions.sort(key=lambda r: (r["bbox_rel"][1], r["bbox_rel"][0]))
    return regions


def ocr_page_with_qwen_api(image_path: str, cfg: OCRConfig) -> Dict[str, Any]:
    from PIL import Image
    img = Image.open(image_path)
    width, height = img.size

    # Compute input size after smart_resize (for diagnostics / reproducibility)
    input_h, input_w = smart_resize(
        height, width,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        factor=cfg.factor,
    )

    t0 = time.time()

    try:
        output = inference_with_api(
            image_path,
            cfg.prompt,
            model_id=cfg.model_id,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels
        )
    except Exception as e:
        raise RuntimeError(f"Qwen API inference failed: {repr(e)}") from e
    # ==========================================

    elapsed_ms = int((time.time() - t0) * 1000)

    # Unified return schema:
    # - if output is str  -> treated as markdown
    # - if output is dict -> expected to contain markdown / blocks
    if isinstance(output, str):
        cleaned = strip_markdown_code_fences(output)
        result = {
            "markdown": cleaned,
            "blocks": None,
        }
    elif isinstance(output, dict):
        raw_md = output.get("markdown") or output.get("text") or ""
        cleaned = strip_markdown_code_fences(raw_md)
        result = {
            "markdown": cleaned,
            "blocks": output.get("blocks"),  # 期望 blocks 里带 bbox、text、type
            "raw": output,                   # 保留原始输出方便追溯
        }
    else:
        result = {
            "markdown": str(output),
            "blocks": None,
            "raw": output,
        }

    result["diagnostics"] = {
        "model": cfg.model_id,
        "prompt": cfg.prompt,
        "min_pixels": cfg.min_pixels,
        "max_pixels": cfg.max_pixels,
        "smart_resize": {"in_w": width, "in_h": height, "out_w": input_w, "out_h": input_h, "factor": cfg.factor},
        "elapsed_ms": elapsed_ms,
    }
    # Provide page image size for downstream normalization
    result["page_image"] = {"width_px": width, "height_px": height}
    return result


def _cache_key(image_sha256: str) -> str:
    # image_sha256 形如 "sha256:abcd..."
    return image_sha256.replace(":", "_") + ".json"


def ocr_document_pages(render_results: List[Dict[str, Any]], cfg: OCRConfig) -> List[Dict[str, Any]]:
    """
    render_results: render manifest from step 1
    (each item contains image_path, image_sha256, page_number, etc.)

    Returns:
      OCR results aligned by page_number.
      Each item contains page_number and success / failure metadata.
    """
    _ensure_dir(cfg.cache_dir)

    all_results: List[Dict[str, Any]] = []

    for r in render_results:
        page_number = int(r["page_number"])
        image_path = r["image_path"]
        image_sha256 = r.get("image_sha256", "")

        cache_path = os.path.join(cfg.cache_dir, _cache_key(image_sha256 or f"page_{page_number:04d}"))

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            all_results.append(cached)
            continue

        last_err: Optional[str] = None
        for attempt in range(cfg.max_retries + 1):
            try:
                ocr = ocr_page_with_qwen_api(image_path, cfg)
                out = {
                    "page_number": page_number,
                    "ok": True,
                    "image_path": image_path,
                    "image_sha256": image_sha256,
                    "markdown": ocr.get("markdown", ""),
                    "blocks": ocr.get("blocks"),
                    "diagnostics": ocr.get("diagnostics", {}),
                    "page_width_px": int(r.get("width_px") or ocr.get("page_image", {}).get("width_px") or 0),
                    "page_height_px": int(r.get("height_px") or ocr.get("page_image", {}).get("height_px") or 0),
                }
                if "raw" in ocr:
                    out["raw"] = ocr["raw"]  # 可选：体积可能大，但追溯强

                # Extract only table/image bboxes (relative coords) from markdown markers
                pw = out.get("page_width_px", 0)
                ph = out.get("page_height_px", 0)
                out["media_regions"] = extract_media_regions_from_markdown(out.get("markdown", ""), pw, ph)

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

                all_results.append(out)
                last_err = None
                break
            except Exception:
                last_err = traceback.format_exc()
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_backoff_sec * (attempt + 1))
                else:
                    out = {
                        "page_number": page_number,
                        "ok": False,
                        "image_path": image_path,
                        "image_sha256": image_sha256,
                        "error": last_err,
                        "diagnostics": {
                            "model": cfg.model_id,
                            "prompt": cfg.prompt,
                            "min_pixels": cfg.min_pixels,
                            "max_pixels": cfg.max_pixels,
                        }
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)

                    all_results.append(out)

        time.sleep(cfg.sleep_between_pages_sec)

    # Sort by page_number to ensure stable order
    all_results.sort(key=lambda x: x["page_number"])
    return all_results


def attach_ocr_results(doc: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Attach OCR results back to doc.json:
      - text_raw, text_source
      - spans (only when regions exist)
      - merge diagnostics
      - append errors / warnings
    """
    page_map = {p["page_number"]: p for p in doc.get("pages", [])}

    for res in ocr_results:
        pn = res["page_number"]
        page = page_map.get(pn)
        if not page:
            doc.setdefault("warnings", []).append({
                "stage": "attach_ocr",
                "message": f"page {pn} not found in doc.pages"
            })
            continue

        if res.get("ok"):
            page["text_raw"] = res.get("markdown", "")  # Use cleaned markdown as the canonical text_raw
            page["text_source"] = "ocr"

            # If markdown contains table/image markers, attach only those regions as spans with RELATIVE bboxes.
            media = res.get("media_regions")
            spans: List[Dict[str, Any]] = []
            if isinstance(media, list) and len(media) > 0:
                for i, m in enumerate(media, start=1):
                    spans.append({
                        "span_id": f"{pn}:ocr:region:{i:04d}",
                        "type": m.get("type", "region"),  # "table" or "image"
                        "bbox_rel": m.get("bbox_rel"),
                        # keep pixel bbox for debugging/visualization if needed
                        "bbox_px": m.get("bbox_px"),
                        "source": "ocr",
                    })
                page["spans"] = spans
            else:
                # Pure-text page: keep content in text_raw only
                page["spans"] = []

            # Merge diagnostics
            page.setdefault("diagnostics", {})
            page["diagnostics"]["ocr"] = res.get("diagnostics", {})
            page["diagnostics"]["ocr"]["image_sha256"] = res.get("image_sha256")
        else:
            # Failure case: record error without blocking the entire document
            doc.setdefault("errors", []).append({
                "stage": "ocr",
                "page_number": pn,
                "message": res.get("error", "unknown error")
            })
            page.setdefault("diagnostics", {})
            page["diagnostics"]["ocr"] = res.get("diagnostics", {})

    return doc

def process_document(doc: Dict[str, Any], cfg: OCRConfig) -> Dict[str, Any]:
    """Run OCR for a full document dict and attach results.

    This function is intentionally *wrapper-preserving*: it updates `doc.pages[*]`
    in place and returns the same document object with all top-level fields kept
    (e.g., doc_id/source/num_pages/metadata/...)

    Expected input schema:
      doc = {
        "doc_id": ...,
        "source": ...,
        "num_pages": ...,
        "metadata": ...,
        "pages": [ {"page_number":..., "image_path":..., ...}, ...]
      }

    Returns:
      The updated document dict with OCR fields merged into each page.
    """
    pages = doc.get("pages")
    if not isinstance(pages, list):
        raise ValueError("doc['pages'] must be a list")

    ocr_results = ocr_document_pages(pages, cfg)
    return attach_ocr_results(doc, ocr_results)


def process_document_file(input_json_path: str, output_json_path: str, cfg: OCRConfig) -> Dict[str, Any]:
    """Load a document JSON, run OCR, and write the full updated JSON back.

    This is a convenience wrapper to avoid accidentally writing only `pages`.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    updated = process_document(doc, cfg)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_json_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    return updated
