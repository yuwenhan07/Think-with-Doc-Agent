from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "1-pdf_parsing"))
sys.path.append(str(ROOT / "2-VLM_understanding"))
sys.path.append(str(ROOT / "3-embedding"))
sys.path.append(str(ROOT / "4-Planner"))

from pdf_render import render_pdf_to_document, compute_doc_id  # type: ignore  # noqa: E402
from ocr import OCRConfig, process_document_file  # type: ignore  # noqa: E402
from summary import summarize_doc_json  # type: ignore  # noqa: E402
from chunk_output import demo_build_blocks_for_doc_json  # type: ignore  # noqa: E402
from embedding import build_indexes  # type: ignore  # noqa: E402
from executor import BudgetConfig, Executor  # type: ignore  # noqa: E402
from planners import PlannerConfig  # type: ignore  # noqa: E402
from skills import LLMConfig  # type: ignore  # noqa: E402


@dataclass
class PipelineArtifacts:
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

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "PipelineArtifacts":
        return cls(
            run_dir=Path(data["run_dir"]),
            pdf_path=Path(data["pdf_path"]),
            index_dir=Path(data["index_dir"]),
            asset_base_dir=Path(data["asset_base_dir"]),
            chunks_json=Path(data["chunks_json"]),
            doc_id=data["doc_id"],
            doc_name=data["doc_name"],
            num_pages=int(data["num_pages"]),
        )


def _default_logger(msg: str) -> None:
    print(msg, flush=True)


def _require_qianfan_key() -> None:
    if not os.environ.get("QianFan_API_KEY"):
        raise RuntimeError("Missing env var QianFan_API_KEY for OCR/summary/embedding calls.")


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifacts_path(run_dir: Path) -> Path:
    return run_dir / "artifacts.json"


def _load_cached_artifacts(run_dir: Path) -> Optional[PipelineArtifacts]:
    artifacts_file = _artifacts_path(run_dir)
    if not artifacts_file.exists():
        return None
    data = json.loads(artifacts_file.read_text(encoding="utf-8"))
    artifacts = PipelineArtifacts.from_dict(data)
    required = [
        artifacts.index_dir / "summary.index.faiss",
        artifacts.index_dir / "summary.meta.jsonl",
        artifacts.index_dir / "blocks.manifest.json",
        artifacts.chunks_json,
        artifacts.asset_base_dir,
    ]
    for path in required:
        if not path.exists():
            return None
    return artifacts


def _save_artifacts(artifacts: PipelineArtifacts) -> None:
    _write_json(_artifacts_path(artifacts.run_dir), artifacts.to_dict())


def build_pipeline(
    pdf_path: str,
    *,
    run_root: Optional[Path] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> PipelineArtifacts:
    log = logger or _default_logger
    _require_qianfan_key()

    src_pdf = Path(pdf_path)
    if not src_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {src_pdf}")

    run_root = run_root or (ROOT / "visulaize" / "runs")
    run_root.mkdir(parents=True, exist_ok=True)

    doc_id = compute_doc_id(str(src_pdf))
    doc_hash = doc_id.split(":")[-1]
    run_dir = run_root / doc_hash
    cached = _load_cached_artifacts(run_dir)
    if cached:
        log(f"Cache hit. Using existing artifacts in: {run_dir}")
        return cached

    run_dir.mkdir(parents=True, exist_ok=True)

    doc_dir = run_dir / "doc"
    imgs_dir = run_dir / "imgs"
    json_dir = run_dir / "jsons"
    chunks_dir = run_dir / "chunks"
    index_dir = run_dir / "index_out"

    doc_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    doc_name = src_pdf.name
    pdf_copy = doc_dir / doc_name
    shutil.copy2(src_pdf, pdf_copy)

    log("Step 1/5: Render PDF to page images...")
    render_doc = render_pdf_to_document(str(pdf_copy), str(imgs_dir), dpi=144, overwrite=True)
    render_json = json_dir / f"{pdf_copy.stem}_render.json"
    _write_json(render_json, render_doc)

    log("Step 2/5: OCR pages with Qwen-VL...")
    ocr_json = json_dir / f"{pdf_copy.stem}_ocr.json"
    process_document_file(str(render_json), str(ocr_json), OCRConfig())

    log("Step 3/5: Summarize pages...")
    summary_json = json_dir / f"{pdf_copy.stem}_ocr_with_summary.json"
    summarize_doc_json(str(ocr_json), str(summary_json))

    log("Step 4/5: Build text/figure/table blocks...")
    chunks_json = json_dir / f"{pdf_copy.stem}_ocr_with_summary_chunks.json"
    assets_root = chunks_dir / pdf_copy.stem
    demo_build_blocks_for_doc_json(
        input_json_path=str(summary_json),
        output_json_path=str(chunks_json),
        assets_root=str(assets_root),
    )

    log("Step 5/5: Build FAISS indexes...")
    build_indexes(str(chunks_json), str(index_dir))

    num_pages = int(render_doc.get("num_pages") or 0)
    log("Pipeline complete.")

    artifacts = PipelineArtifacts(
        run_dir=run_dir,
        pdf_path=pdf_copy,
        index_dir=index_dir,
        asset_base_dir=chunks_dir,
        chunks_json=chunks_json,
        doc_id=doc_id,
        doc_name=doc_name,
        num_pages=num_pages,
    )
    _save_artifacts(artifacts)
    return artifacts


def run_query(
    query: str,
    artifacts: PipelineArtifacts,
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    log = logger or _default_logger
    _require_qianfan_key()

    planner_cfg = PlannerConfig()
    llm_cfg = LLMConfig()
    budget = BudgetConfig()

    log("Running planner/executor...")
    executor = Executor(
        index_dir=artifacts.index_dir,
        asset_base_dir=artifacts.asset_base_dir,
        planner_config=planner_cfg,
        llm_config=llm_cfg,
        budget=budget,
    )
    result = executor.run(query)
    log("Planner/executor finished.")
    return result
