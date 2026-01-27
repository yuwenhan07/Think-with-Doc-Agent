import os
# Use spawn to avoid fork+Cuda exit crashes when vLLM uses multiprocessing.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
import faiss

from PIL import Image
from vllm import LLM

VLLM_MODEL = "/home/work/models/Qwen3-VL-Embedding-2B"
VLLM_RUNNER = "pooling"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VLLM_CLIENT: Optional[LLM] = None
VLLM_TP = int(os.environ.get("VLLM_TP_SIZE", "4"))


def _get_vllm_client() -> LLM:
    global _VLLM_CLIENT
    if _VLLM_CLIENT is None:
        _VLLM_CLIENT = LLM(
            model=VLLM_MODEL,
            runner=VLLM_RUNNER,
            tensor_parallel_size=VLLM_TP,
        )
    return _VLLM_CLIENT


def shutdown_vllm_client() -> None:
    """Cleanly stop vLLM engine processes to avoid core dumps on exit."""
    global _VLLM_CLIENT
    if _VLLM_CLIENT is None:
        return
    try:
        # vLLM does not expose a public close() yet; use engine_core shutdown.
        _VLLM_CLIENT.llm_engine.engine_core.shutdown()
    finally:
        _VLLM_CLIENT = None


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec) + 1e-12
    return vec / denom


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _embed_inputs(inputs: list[dict]) -> List[np.ndarray]:
    if not inputs:
        return []

    outputs = _get_vllm_client().embed(inputs)
    out: List[np.ndarray] = []
    for item in outputs:
        out.append(np.array(item.outputs.embedding, dtype=np.float32))
    if not out:
        raise RuntimeError("Empty embedding response from vLLM.")
    return out


def embed_text(text: str) -> np.ndarray:
    v = _embed_inputs([{"prompt": text}])[0]
    return l2_normalize(v.astype(np.float32))


def embed_image(image_path: str) -> np.ndarray:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(p).convert("RGB")
    v = _embed_inputs([
        {"prompt": IMAGE_PLACEHOLDER, "multi_modal_data": {"image": image}}
    ])[0]
    return l2_normalize(v.astype(np.float32))


def resolve_asset_path(asset_path: Optional[str], base_dir: str) -> Optional[str]:
    """
    asset_path 可能是相对路径（如 ../chunks/...png）。
    base_dir 建议传 index_out 的父目录或工程根目录。
    """
    if not asset_path:
        return None
    p = Path(asset_path)
    if p.is_absolute():
        return str(p)
    # 相对路径：以 base_dir 为起点拼接
    return str((Path(base_dir) / p).resolve())


def faiss_search(index: faiss.Index, query_vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    # query_vec: [d]
    D, I = index.search(query_vec.reshape(1, -1).astype(np.float32), topk)
    return D[0], I[0]


def load_page_block_shard(index_dir: Path, page_no: int) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load per-page block shard: blocks.pXXXX.index.faiss + blocks.pXXXX.meta.jsonl
    """
    idx_path = index_dir / f"blocks.p{page_no:04d}.index.faiss"
    meta_path = index_dir / f"blocks.p{page_no:04d}.meta.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing block shard for page {page_no}: {idx_path} / {meta_path}")
    return faiss.read_index(str(idx_path)), load_jsonl(str(meta_path))


def search_blocks_in_pages(
    qv: np.ndarray,
    index_dir: Path,
    page_nos: Iterable[int],
    blocks_topk: int,
    final_topk: int,
    asset_base_dir: str = ".",
) -> List[Dict[str, Any]]:
    """
    Search only selected page shards and merge results by score.
    Assumes qv is already L2-normalized.
    """
    merged: List[Dict[str, Any]] = []

    for page_no in page_nos:
        try:
            page_index, page_meta = load_page_block_shard(index_dir, page_no)
        except FileNotFoundError:
            # If a page has no blocks shard, skip it (manifest should prevent this).
            continue

        D, I = faiss_search(page_index, qv, blocks_topk)
        for score, idx in zip(D, I):
            if idx < 0 or idx >= len(page_meta):
                continue
            item = dict(page_meta[idx])
            # Resolve asset_path for non-text blocks for convenience
            if item.get("type") != "text":
                item["asset_path"] = resolve_asset_path(item.get("asset_path"), asset_base_dir)
            merged.append(pretty_hit(item, float(score)))

    # Merge-sort by score desc (IndexFlatIP with L2-normalized vectors => cosine similarity)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:final_topk]


def pretty_hit(item: Dict[str, Any], score: float) -> Dict[str, Any]:
    # 精简输出字段，保留回链所需信息
    out = {
        "score": float(score),
        "page_number": item.get("page_number"),
        "id": item.get("id"),
        "block_id": item.get("block_id"),
        "type": item.get("type"),
    }
    if item.get("type") == "text":
        out["text"] = item.get("text")
    else:
        out["asset_path"] = item.get("asset_path")
        out["text"] = item.get("text")  # caption 若有
        out["bbox_px"] = item.get("bbox_px")
    return out


def search_text_two_stage(
    query: str,
    index_dir: Path,
    summary_index: faiss.Index,
    summary_meta: List[Dict[str, Any]],
    summary_topk: int = 5,
    blocks_topk: int = 50,
    final_topk: int = 10,
    asset_base_dir: str = ".",
) -> Dict[str, Any]:
    """
    Two-stage retrieval with per-page block shards:
      1) Search summary index to get candidate pages
      2) Search only those pages' block shards and merge results
    """
    qv = embed_text(query)  # already L2-normalized in embed_text()

    # stage 1: summary
    sD, sI = faiss_search(summary_index, qv, summary_topk)
    candidate_pages: List[int] = []
    summary_hits: List[Dict[str, Any]] = []

    for score, idx in zip(sD, sI):
        if idx < 0 or idx >= len(summary_meta):
            continue
        item = summary_meta[idx]
        page_no = item["page_number"]
        candidate_pages.append(page_no)
        summary_hits.append({
            "score": float(score),
            "page_number": page_no,
            "id": item["id"],
            "summary": item.get("text"),
        })

    # de-dup but keep order (Python 3.7+ preserves dict insertion order)
    candidate_pages = list(dict.fromkeys(candidate_pages))

    # stage 2: blocks (search only candidate pages)
    block_hits = search_blocks_in_pages(
        qv=qv,
        index_dir=index_dir,
        page_nos=candidate_pages,
        blocks_topk=blocks_topk,
        final_topk=final_topk,
        asset_base_dir=asset_base_dir,
    )

    return {
        "query": query,
        "candidate_pages": candidate_pages,
        "summary_hits": summary_hits,
        "block_hits": block_hits,
    }


def search_image_blocks(
    image_path: str,
    index_dir: Path,
    manifest: Dict[str, Any],
    topk: int = 10,
    asset_base_dir: str = ".",
    restrict_pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Image retrieval over per-page block shards.
    If restrict_pages is provided, only those pages are searched; otherwise all pages in manifest.
    """
    qv = embed_image(image_path)  # already L2-normalized

    if restrict_pages is not None:
        page_nos = restrict_pages
    else:
        page_nos = [p["page_number"] for p in manifest.get("pages", [])]

    block_hits = search_blocks_in_pages(
        qv=qv,
        index_dir=index_dir,
        page_nos=page_nos,
        blocks_topk=max(topk, 50),  # a bit wider per-shard, then global topk
        final_topk=topk,
        asset_base_dir=asset_base_dir,
    )

    return {
        "query_image": str(Path(image_path).resolve()),
        "searched_pages": page_nos,
        "block_hits": block_hits,
    }
