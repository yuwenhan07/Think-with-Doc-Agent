from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "1-pdf_parsing"))
sys.path.append(str(ROOT / "2-VLM_understanding"))
sys.path.append(str(ROOT / "3-embedding"))

from pdf_render import render_pdf_to_document, compute_doc_id  # type: ignore  # noqa: E402
from ocr import OCRConfig, process_document_file  # type: ignore  # noqa: E402
from summary import summarize_doc_json  # type: ignore  # noqa: E402
from chunk_output import demo_build_blocks_for_doc_json  # type: ignore  # noqa: E402
from embedding import build_indexes  # type: ignore  # noqa: E402

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None  # type: ignore


@dataclass
class RunArtifacts:
    run_dir: Path
    pdf_path: Path
    index_dir: Path
    asset_base_dir: Path
    chunks_json: Path
    doc_id: str
    doc_name: str
    num_pages: int

    def to_dict(self) -> Dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "pdf_path": str(self.pdf_path),
            "index_dir": str(self.index_dir),
            "asset_base_dir": str(self.asset_base_dir),
            "chunks_json": str(self.chunks_json),
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "num_pages": str(self.num_pages),
        }


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_paths_ok(run_dir: Path) -> bool:
    artifacts_file = run_dir / "artifacts.json"
    if not artifacts_file.exists():
        return False
    data = json.loads(artifacts_file.read_text(encoding="utf-8"))
    required = [
        Path(data["index_dir"]) / "summary.index.faiss",
        Path(data["index_dir"]) / "summary.meta.jsonl",
        Path(data["index_dir"]) / "blocks.manifest.json",
        Path(data["chunks_json"]),
        Path(data["asset_base_dir"]),
    ]
    return all(p.exists() for p in required)


def _iter_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])


def process_one_pdf(
    pdf_path: Path,
    *,
    out_root: Path,
    dpi: int,
    overwrite: bool,
) -> Dict[str, object]:
    t0 = time.time()
    doc_id = compute_doc_id(str(pdf_path))
    doc_hash = doc_id.split(":")[-1]
    run_dir = out_root / doc_hash

    if not overwrite and _artifact_paths_ok(run_dir):
        return {"pdf": str(pdf_path), "status": "skipped", "run_dir": str(run_dir)}

    run_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = run_dir / "doc"
    imgs_dir = run_dir / "imgs"
    json_dir = run_dir / "jsons"
    chunks_dir = run_dir / "chunks"
    index_dir = run_dir / "index_out"

    for d in (doc_dir, imgs_dir, json_dir, chunks_dir, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    pdf_copy = doc_dir / pdf_path.name
    shutil.copy2(pdf_path, pdf_copy)

    render_doc = render_pdf_to_document(str(pdf_copy), str(imgs_dir), dpi=dpi, overwrite=True)
    render_json = json_dir / f"{pdf_copy.stem}_render.json"
    _write_json(render_json, render_doc)

    ocr_json = json_dir / f"{pdf_copy.stem}_ocr.json"
    process_document_file(str(render_json), str(ocr_json), OCRConfig())

    summary_json = json_dir / f"{pdf_copy.stem}_ocr_with_summary.json"
    summarize_doc_json(str(ocr_json), str(summary_json))

    chunks_json = json_dir / f"{pdf_copy.stem}_ocr_with_summary_chunks.json"
    assets_root = chunks_dir / pdf_copy.stem
    demo_build_blocks_for_doc_json(
        input_json_path=str(summary_json),
        output_json_path=str(chunks_json),
        assets_root=str(assets_root),
    )

    build_indexes(str(chunks_json), str(index_dir))

    artifacts = RunArtifacts(
        run_dir=run_dir,
        pdf_path=pdf_copy,
        index_dir=index_dir,
        asset_base_dir=chunks_dir,
        chunks_json=chunks_json,
        doc_id=doc_id,
        doc_name=pdf_copy.name,
        num_pages=int(render_doc.get("num_pages") or 0),
    )
    _write_json(run_dir / "artifacts.json", artifacts.to_dict())

    return {
        "pdf": str(pdf_path),
        "status": "ok",
        "run_dir": str(run_dir),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def _run_with_progress(tasks: Iterable[Path], total: int, worker_fn):
    if tqdm is None:
        done = 0
        for item in tasks:
            yield worker_fn(item)
            done += 1
            print(f"[{done}/{total}] done: {item.name}", flush=True)
    else:
        for item in tqdm(tasks, total=total):
            yield worker_fn(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="/yuwenhan/Proj/think-with-doc/data/benchmarks/MMLongBench/documents",
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(ROOT / "Evaluation" / "runs"),
        help="Output root for all artifacts.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF dir not found: {pdf_dir}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs = _iter_pdfs(pdf_dir)
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    results: List[Dict[str, object]] = []

    if args.workers <= 1:
        for res in _run_with_progress(pdfs, len(pdfs), lambda p: process_one_pdf(p, out_root=out_root, dpi=args.dpi, overwrite=args.overwrite)):
            results.append(res)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _submit(p: Path):
            return process_one_pdf(p, out_root=out_root, dpi=args.dpi, overwrite=args.overwrite)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_submit, p): p for p in pdfs}
            if tqdm is None:
                done = 0
                for fut in as_completed(futures):
                    results.append(fut.result())
                    done += 1
                    print(f"[{done}/{len(pdfs)}] done: {futures[fut].name}", flush=True)
            else:
                for fut in tqdm(as_completed(futures), total=len(pdfs)):
                    results.append(fut.result())

    _write_json(out_root / "batch_results.json", {"results": results})
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"Done. ok={ok} skipped={skipped} total={len(results)}")


if __name__ == "__main__":
    main()
