from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "1-pdf_parsing"))
sys.path.append(str(ROOT / "4-Planner"))

from pdf_render import compute_doc_id  # type: ignore  # noqa: E402
from executor import BudgetConfig, Executor  # type: ignore  # noqa: E402
from planners import PlannerConfig  # type: ignore  # noqa: E402
from skills import LLMConfig  # type: ignore  # noqa: E402


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_pdf(root_pdf: Path, filename: str) -> Optional[Path]:
    direct = root_pdf / filename
    if direct.exists():
        return direct
    matches = [p for p in root_pdf.rglob(filename) if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


def _load_artifacts(run_dir: Path) -> Optional[Dict[str, str]]:
    artifacts_path = run_dir / "artifacts.json"
    if not artifacts_path.exists():
        return None
    data = json.loads(artifacts_path.read_text(encoding="utf-8"))
    return {
        "artifacts_path": str(artifacts_path),
        "index_dir": data.get("index_dir"),
        "asset_base_dir": data.get("asset_base_dir"),
        "chunks_json": data.get("chunks_json"),
        "doc_id": data.get("doc_id"),
        "doc_name": data.get("doc_name"),
    }


def _build_executor_cache() -> Dict[str, Executor]:
    return {}


def _get_executor(
    cache: Dict[str, Executor],
    index_dir: Path,
    asset_base_dir: Path,
    budget: BudgetConfig,
    planner_cfg: PlannerConfig,
    llm_cfg: LLMConfig,
) -> Executor:
    key = f"{index_dir}::{asset_base_dir}"
    if key in cache:
        return cache[key]
    executor = Executor(
        index_dir=index_dir,
        asset_base_dir=asset_base_dir,
        planner_config=planner_cfg,
        llm_config=llm_cfg,
        budget=budget,
    )
    cache[key] = executor
    return executor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--root_pdf", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default=str(ROOT / "Evaluation" / "runs"))
    parser.add_argument("--query_field", type=str, default="question")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--max_search_calls", type=int, default=3)
    parser.add_argument("--max_rewrite_calls", type=int, default=2)
    parser.add_argument("--max_blocks_context", type=int, default=8)
    args = parser.parse_args()

    items = _read_json(Path(args.input_json))
    if not isinstance(items, list):
        raise ValueError("input_json must be a list of records.")

    root_pdf = Path(args.root_pdf)
    runs_root = Path(args.runs_root)
    planner_cfg = PlannerConfig()
    llm_cfg = LLMConfig()
    budget = BudgetConfig(
        max_turns=args.max_turns,
        max_search_calls=args.max_search_calls,
        max_rewrite_calls=args.max_rewrite_calls,
        max_blocks_context=args.max_blocks_context,
    )

    cache = _build_executor_cache()
    results: List[Dict[str, object]] = []
    limit = args.limit if args.limit and args.limit > 0 else len(items)

    for item in items[:limit]:
        if not isinstance(item, dict):
            results.append({"error": "invalid_item_type"})
            continue
        filename = item.get("doc_id")
        query = item.get(args.query_field)
        uuid = item.get("uuid")

        if not filename or not query:
            results.append({
                "uuid": uuid,
                "doc_id": filename,
                "query": query,
                "error": "missing_doc_id_or_query",
            })
            continue

        pdf_path = _find_pdf(root_pdf, str(filename))
        if not pdf_path:
            results.append({
                "uuid": uuid,
                "doc_id": filename,
                "query": query,
                "error": "pdf_not_found",
            })
            continue

        doc_id = compute_doc_id(str(pdf_path))
        doc_hash = doc_id.split(":")[-1]
        run_dir = runs_root / doc_hash
        artifacts = _load_artifacts(run_dir)
        if not artifacts:
            results.append({
                "uuid": uuid,
                "doc_id": filename,
                "query": query,
                "pdf_path": str(pdf_path),
                "doc_hash": doc_hash,
                "error": "artifacts_not_found",
            })
            continue

        try:
            executor = _get_executor(
                cache,
                index_dir=Path(artifacts["index_dir"]),
                asset_base_dir=Path(artifacts["asset_base_dir"]),
                budget=budget,
                planner_cfg=planner_cfg,
                llm_cfg=llm_cfg,
            )
            result = executor.run(str(query))
            results.append({
                "uuid": uuid,
                "doc_id": filename,
                "query": query,
                "pdf_path": str(pdf_path),
                "doc_hash": doc_hash,
                "artifacts": artifacts,
                "final": result.get("final"),
                "trace": result.get("trace", []),
            })
        except Exception as exc:
            results.append({
                "uuid": uuid,
                "doc_id": filename,
                "query": query,
                "pdf_path": str(pdf_path),
                "doc_hash": doc_hash,
                "artifacts": artifacts,
                "error": str(exc),
            })

    payload = {
        "input_json": str(Path(args.input_json)),
        "root_pdf": str(root_pdf),
        "runs_root": str(runs_root),
        "query_field": args.query_field,
        "planner_config": asdict(planner_cfg),
        "llm_config": asdict(llm_cfg),
        "budget": asdict(budget),
        "results": results,
    }
    _write_json(Path(args.output), payload)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
