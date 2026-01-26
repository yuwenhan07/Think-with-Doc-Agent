"""
Standalone quick test for search.py

Usage:
  export QianFan_API_KEY="bce-v3/..."
  python test_search.py --index_dir index_out --asset_base_dir .. --query "What is the meaning of life?"
  python test_search.py --index_dir index_out --asset_base_dir .. --query_image ../chunks/2310.08560v2/page_0002.png

demo:
  python test_search.py --query "What is the meaning of Memgpt?"
  python test_search.py  --query_image ../demo/chunks/2310.08560v2/page_0002/p0002_2-ocr-region-0001.png 
"""

import argparse
import json
from pathlib import Path

import faiss

from search import (
    load_jsonl,
    load_json,
    search_text_two_stage,
    search_image_blocks,
    shutdown_vllm_client,
)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="../demo/index_out/2310.08560v2", help="Directory containing summary index + per-page block shards")
    parser.add_argument("--asset_base_dir", type=str, default="../demo/chunks/2310.08560v2", help="Base dir to resolve relative asset_path")

    parser.add_argument("--query", type=str, default=None, help="Text query (two-stage: summary -> per-page blocks)")
    parser.add_argument("--query_image", type=str, default=None, help="Image query path (optional)")

    parser.add_argument("--summary_topk", type=int, default=5)
    parser.add_argument("--blocks_topk", type=int, default=20)
    parser.add_argument("--final_topk", type=int, default=10)

    args = parser.parse_args()
    index_dir = Path(args.index_dir)

    # Load summary index + metadata
    summary_index = faiss.read_index(str(index_dir / "summary.index.faiss"))
    summary_meta = load_jsonl(str(index_dir / "summary.meta.jsonl"))

    # Load blocks manifest (lists which pages have shards)
    manifest_path = index_dir / "blocks.manifest.json"
    if manifest_path.exists():
        manifest = load_json(str(manifest_path))
    else:
        manifest = {"pages": []}

    did_any = False

    try:
        if args.query:
            did_any = True
            result = search_text_two_stage(
                query=args.query,
                index_dir=index_dir,
                summary_index=summary_index,
                summary_meta=summary_meta,
                summary_topk=args.summary_topk,
                blocks_topk=args.blocks_topk,
                final_topk=args.final_topk,
                asset_base_dir=args.asset_base_dir,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        if args.query_image:
            did_any = True
            result = search_image_blocks(
                image_path=args.query_image,
                index_dir=index_dir,
                manifest=manifest,
                topk=args.final_topk,
                asset_base_dir=args.asset_base_dir,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        if not did_any:
            print("Nothing to do. Provide --query or --query_image.")
    finally:
        shutdown_vllm_client()


if __name__ == "__main__":
    main()
