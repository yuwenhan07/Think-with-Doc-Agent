import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import faiss

from PIL import Image
import atexit

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from vllm import LLM

VLLM_MODEL = "/home/work/bos-qgq/wh/models/Qwen3-VL-Embedding-2B"
VLLM_RUNNER = "pooling"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VLLM_CLIENT: Optional[LLM] = None
VLLM_TP = int(os.environ.get("VLLM_TP_SIZE", "4"))

_FIGURE_RE = re.compile(r"\bfig(?:ure)?\.?\s*([0-9]+[a-z]?)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\btable\.?\s*([0-9]+[a-z]?)\b", re.IGNORECASE)
_HEADING_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")
_SECTION_RE = re.compile(r"\bsection\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)

def _get_vllm_client() -> LLM:
    global _VLLM_CLIENT
    if _VLLM_CLIENT is None:
        _VLLM_CLIENT = LLM(
            model=VLLM_MODEL,
            runner=VLLM_RUNNER,
            tensor_parallel_size=VLLM_TP,
        )
    return _VLLM_CLIENT


def _try_shutdown_engine(obj: Any) -> None:
    if obj is None:
        return
    for name in ("shutdown", "stop", "close"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


def shutdown_vllm() -> None:
    global _VLLM_CLIENT
    client = _VLLM_CLIENT
    _VLLM_CLIENT = None
    if client is None:
        return
    _try_shutdown_engine(client)
    _try_shutdown_engine(getattr(client, "llm_engine", None))
    _try_shutdown_engine(getattr(client, "engine", None))


atexit.register(shutdown_vllm)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec) + 1e-12
    return vec / denom

def _embed_inputs(inputs: list[dict]) -> list[np.ndarray]:
    """
    inputs: vLLM embedding inputs with "prompt" and optional "multi_modal_data".
    return: [np.ndarray(d), ...]
    """
    if not inputs:
        return []

    outputs = _get_vllm_client().embed(inputs)
    out: list[np.ndarray] = []
    for item in outputs:
        emb = np.array(item.outputs.embedding, dtype=np.float32)
        out.append(emb)
    if not out:
        raise RuntimeError("Empty embedding response from vLLM.")
    return out

def embed_text(text: str) -> np.ndarray:
    return _embed_inputs([{"prompt": text}])[0]

def embed_image(image_path: str) -> np.ndarray:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(p).convert("RGB")
    return _embed_inputs([
        {"prompt": IMAGE_PLACEHOLDER, "multi_modal_data": {"image": image}}
    ])[0]

def build_faiss_cosine(vectors: list[np.ndarray]) -> faiss.Index:
    # cosine = inner product after L2 normalize
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    mat = np.vstack([l2_normalize(v) for v in vectors]).astype(np.float32)
    index.add(mat)
    return index

def iter_summary_items(doc: dict):
    doc_id = doc.get("doc_id")
    for p in doc["pages"]:
        s = p.get("page_summary")
        if not s:
            continue
        yield {
            "id": f"{doc_id}:p{p['page_number']}:summary",
            "doc_id": doc_id,
            "page_number": p["page_number"],
            "text": s,
        }

def iter_block_items(doc: dict):
    doc_id = doc.get("doc_id")
    for p in doc["pages"]:
        page_no = p["page_number"]
        for b in p.get("blocks", []):
            block_id = b.get("block_id")
            btype = b.get("type")
            item = {
                "id": f"{doc_id}:{block_id}" if block_id else f"{doc_id}:p{page_no}:unknown",
                "doc_id": doc_id,
                "page_number": page_no,
                "block_id": block_id,
                "type": btype,
                "text": b.get("text"),
                "asset_path": b.get("asset_path"),
                "bbox_px": b.get("bbox_px"),
                "crop_work_size": b.get("crop_work_size"),
                "span_id": b.get("span_id"),
                "source": b.get("source"),
            }
            if not item["text"] and not item["asset_path"]:
                continue
            yield item

def iter_block_items_by_page(doc: dict):
    """
    Group block items by page_number.
    Returns: dict[int, list[item]]
    """
    by_page = defaultdict(list)
    for it in iter_block_items(doc):
        by_page[it["page_number"]].append(it)
    return by_page


def _extract_headings(text_raw: str, max_lines: int = 8) -> List[str]:
    if not text_raw:
        return []
    lines = [l.strip() for l in text_raw.splitlines() if l.strip()]
    head_lines = lines[:max_lines]
    headings: List[str] = []
    for line in head_lines:
        if len(line) > 90:
            continue
        if _HEADING_NUM_RE.match(line):
            headings.append(line)
            continue
        if line.isupper() and len(line.split()) <= 10:
            headings.append(line)
            continue
        if re.match(r"^[A-Z][A-Za-z0-9 ,:\-]{3,}$", line) and len(line.split()) <= 10:
            headings.append(line)
    # de-dup while preserving order
    return list(dict.fromkeys(headings))


def _anchor_entry(page_no: int, block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page_number": page_no,
        "block_id": block.get("block_id"),
        "type": block.get("type"),
        "text": block.get("text"),
        "asset_path": block.get("asset_path"),
        "bbox_px": block.get("bbox_px"),
        "span_id": block.get("span_id"),
        "source": block.get("source"),
    }


def _is_caption_like(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("figure") or t.startswith("fig.") or t.startswith("fig ") or t.startswith("table")


def build_locator_index(doc: Dict[str, Any]) -> Dict[str, Any]:
    locator: Dict[str, Any] = {
        "doc_id": doc.get("doc_id"),
        "figures": {},
        "tables": {},
        "sections": {},
        "page_headings": {},
        "pages": {},
    }

    for page in doc.get("pages", []):
        page_no = page.get("page_number")
        text_raw = page.get("text_raw") or ""
        image_path = page.get("image_path")
        if page_no is not None and image_path:
            snippet = re.sub(r"\s+", " ", text_raw).strip()
            locator["pages"][str(page_no)] = {
                "image_path": image_path,
                "text_snippet": snippet[:500],
            }
        headings = _extract_headings(text_raw)
        if headings:
            locator["page_headings"][str(page_no)] = headings
            for h in headings:
                m = _HEADING_NUM_RE.match(h)
                if m:
                    label = m.group(1)
                    locator["sections"].setdefault(label, []).append({
                        "page_number": page_no,
                        "heading": h,
                    })

        for match in _SECTION_RE.finditer(text_raw):
            label = match.group(1)
            locator["sections"].setdefault(label, []).append({
                "page_number": page_no,
                "heading": match.group(0),
            })

        for block in page.get("blocks", []) or []:
            text = block.get("text") or ""
            if text and not _is_caption_like(text) and block.get("type") not in ("figure", "table"):
                continue
            for m in _FIGURE_RE.finditer(text):
                label = m.group(1)
                locator["figures"].setdefault(label, []).append(_anchor_entry(page_no, block))
            for m in _TABLE_RE.finditer(text):
                label = m.group(1)
                locator["tables"].setdefault(label, []).append(_anchor_entry(page_no, block))

    return locator

def build_indexes(input_json: str, out_dir: str):
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = json.loads(Path(input_json).read_text(encoding="utf-8"))

        # -------- summary index --------
        sum_items, sum_vecs = [], []
        for it in iter_summary_items(doc):
            v = embed_text(it["text"])
            sum_items.append(it)
            sum_vecs.append(v)

        sum_index = build_faiss_cosine(sum_vecs)
        faiss.write_index(sum_index, str(out_dir / "summary.index.faiss"))
        with open(out_dir / "summary.meta.jsonl", "w", encoding="utf-8") as f:
            for it in sum_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        # -------- block index (per-page shards) --------
        blocks_by_page = iter_block_items_by_page(doc)

        manifest = {
            "doc_id": doc.get("doc_id"),
            "model": VLLM_MODEL,
            "pages": []
        }

        for page_no in sorted(blocks_by_page.keys()):
            page_items = blocks_by_page[page_no]

            blk_items, blk_vecs = [], []
            for it in page_items:
                t = it.get("type")
                if t == "text":
                    v = embed_text(it["text"] or "")
                elif t in ("figure", "table"):
                    # 你的要求：直接读 crop 后图片
                    v = embed_image(it["asset_path"])
                else:
                    # 兜底：优先 text，否则 image
                    if it.get("text"):
                        v = embed_text(it["text"])
                    else:
                        v = embed_image(it["asset_path"])

                blk_items.append(it)
                blk_vecs.append(v)

            # 该页可能没有可编码内容（理论上已过滤，但稳妥处理）
            if not blk_items:
                continue

            blk_index = build_faiss_cosine(blk_vecs)

            idx_name = f"blocks.p{page_no:04d}.index.faiss"
            meta_name = f"blocks.p{page_no:04d}.meta.jsonl"

            faiss.write_index(blk_index, str(out_dir / idx_name))
            with open(out_dir / meta_name, "w", encoding="utf-8") as f:
                for it in blk_items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

            manifest["pages"].append({
                "page_number": page_no,
                "count": len(blk_items),
                "index": idx_name,
                "meta": meta_name,
            })

        with open(out_dir / "blocks.manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        locator_index = build_locator_index(doc)
        with open(out_dir / "locator.index.json", "w", encoding="utf-8") as f:
            json.dump(locator_index, f, ensure_ascii=False, indent=2)

        total_blocks = sum(p["count"] for p in manifest["pages"])
        print(f"OK: summary={len(sum_items)} blocks={total_blocks} (per-page shards={len(manifest['pages'])})")
    finally:
        shutdown_vllm()
