import json
from pathlib import Path
import numpy as np
import faiss
from collections import defaultdict
from typing import Optional

from PIL import Image
from vllm import LLM

VLLM_MODEL = "/models/Qwen3-VL-Embedding-2B"
VLLM_RUNNER = "pooling"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VLLM_CLIENT: Optional[LLM] = None

def _get_vllm_client() -> LLM:
    global _VLLM_CLIENT
    if _VLLM_CLIENT is None:
        _VLLM_CLIENT = LLM(model=VLLM_MODEL, runner=VLLM_RUNNER)
    return _VLLM_CLIENT

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

def build_indexes(input_json: str, out_dir: str):
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

    total_blocks = sum(p["count"] for p in manifest["pages"])
    print(f"OK: summary={len(sum_items)} blocks={total_blocks} (per-page shards={len(manifest['pages'])})")
