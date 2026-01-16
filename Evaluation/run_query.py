from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "1-pdf_parsing"))
sys.path.append(str(ROOT / "4-Planner"))

from pdf_render import compute_doc_id  # type: ignore  # noqa: E402
from executor import BudgetConfig, Executor  # type: ignore  # noqa: E402
from planners import PlannerConfig  # type: ignore  # noqa: E402
from skills import LLMConfig  # type: ignore  # noqa: E402


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_items(path: Path) -> Iterable[object]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("input_json must be a list of records when using .json.")
    for item in data:
        yield item


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    parser.add_argument("--doc_id", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--extract_output", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--max_search_calls", type=int, default=3)
    parser.add_argument("--max_rewrite_calls", type=int, default=2)
    parser.add_argument("--max_blocks_context", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_json)
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
    out_path = Path(args.output)
    _ensure_parent(out_path)
    extract_path = Path(args.extract_output) if args.extract_output else None
    if extract_path:
        _ensure_parent(extract_path)

    run_meta = {
        "input_json": str(Path(args.input_json)),
        "root_pdf": str(root_pdf),
        "runs_root": str(runs_root),
        "query_field": args.query_field,
        "planner_config": asdict(planner_cfg),
        "llm_config": asdict(llm_cfg),
        "budget": asdict(budget),
    }

    with out_path.open("w", encoding="utf-8") as out_f:
        out_f.write(json.dumps({"_meta": run_meta}, ensure_ascii=False) + "\n")
        out_f.flush()

        extract_f = None
        if extract_path:
            extract_f = extract_path.open("w", encoding="utf-8")

        processed = 0
        limit = args.limit if args.limit and args.limit > 0 else 0
        for item in _iter_items(input_path):
            if limit and processed >= limit:
                break
            processed += 1
            if not isinstance(item, dict):
                if isinstance(item, str):
                    filename = args.doc_id or None
                    query = item
                    uuid = None
                    answer = None
                    answer_format = None
                else:
                    out_f.write(json.dumps({"error": "invalid_item_type"}, ensure_ascii=False) + "\n")
                    out_f.flush()
                    continue
            else:
                filename = item.get("doc_id") or (args.doc_id or None)
                query = item.get(args.query_field)
                if not query and args.query_field != "query":
                    query = item.get("query")
                uuid = item.get("uuid")
                answer = item.get("answer")
                answer_format = item.get("answer_format")
                if answer_format is None:
                    answer_format = item.get("answer-formate")

            if not filename or not query:
                out_f.write(json.dumps({
                    "uuid": uuid,
                    "doc_id": filename,
                    "query": query,
                    "error": "missing_doc_id_or_query",
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                if extract_f:
                    extract_f.write(json.dumps({
                        "uuid": uuid,
                        "question": query,
                        "answer": answer,
                        "answer_format": answer_format,
                        "final_text": None,
                        "error": "missing_doc_id_or_query",
                    }, ensure_ascii=False) + "\n")
                    extract_f.flush()
                continue

            pdf_path = _find_pdf(root_pdf, str(filename))
            if not pdf_path:
                out_f.write(json.dumps({
                    "uuid": uuid,
                    "doc_id": filename,
                    "query": query,
                    "error": "pdf_not_found",
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                if extract_f:
                    extract_f.write(json.dumps({
                        "uuid": uuid,
                        "question": query,
                        "answer": answer,
                        "answer_format": answer_format,
                        "final_text": None,
                        "error": "pdf_not_found",
                    }, ensure_ascii=False) + "\n")
                    extract_f.flush()
                continue

            doc_id = compute_doc_id(str(pdf_path))
            doc_hash = doc_id.split(":")[-1]
            run_dir = runs_root / doc_hash
            artifacts = _load_artifacts(run_dir)
            if not artifacts:
                out_f.write(json.dumps({
                    "uuid": uuid,
                    "doc_id": filename,
                    "query": query,
                    "pdf_path": str(pdf_path),
                    "doc_hash": doc_hash,
                    "error": "artifacts_not_found",
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                if extract_f:
                    extract_f.write(json.dumps({
                        "uuid": uuid,
                        "question": query,
                        "answer": answer,
                        "answer_format": answer_format,
                        "final_text": None,
                        "error": "artifacts_not_found",
                    }, ensure_ascii=False) + "\n")
                    extract_f.flush()
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
                final = result.get("final") or {}
                final_text = final.get("final_text") if isinstance(final, dict) else None
                out_f.write(json.dumps({
                    "uuid": uuid,
                    "doc_id": filename,
                    "query": query,
                    "pdf_path": str(pdf_path),
                    "doc_hash": doc_hash,
                    "artifacts": artifacts,
                    "final": result.get("final"),
                    "trace": result.get("trace", []),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                if extract_f:
                    extract_f.write(json.dumps({
                        "uuid": uuid,
                        "question": query,
                        "answer": answer,
                        "answer_format": answer_format,
                        "final_text": final_text,
                    }, ensure_ascii=False) + "\n")
                    extract_f.flush()
            except Exception as exc:
                out_f.write(json.dumps({
                    "uuid": uuid,
                    "doc_id": filename,
                    "query": query,
                    "pdf_path": str(pdf_path),
                    "doc_hash": doc_hash,
                    "artifacts": artifacts,
                    "error": str(exc),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                if extract_f:
                    extract_f.write(json.dumps({
                        "uuid": uuid,
                        "question": query,
                        "answer": answer,
                        "answer_format": answer_format,
                        "final_text": None,
                        "error": str(exc),
                    }, ensure_ascii=False) + "\n")
                    extract_f.flush()

        if extract_f:
            extract_f.close()

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
