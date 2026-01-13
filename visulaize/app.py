from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from pipeline import PipelineArtifacts, build_pipeline, run_query


def _join_logs(lines: List[str]) -> str:
    return "\n".join(lines)


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def process_pdf(pdf_file) -> Tuple[str, Optional[Dict[str, str]]]:
    if pdf_file is None:
        return "Please upload a PDF file.", None

    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    try:
        artifacts = build_pipeline(pdf_file.name, logger=log)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Pipeline failed: {exc}")
        return _join_logs(logs), None

    logs.append(
        f"Ready. doc_id={artifacts.doc_id}, pages={artifacts.num_pages}, run_dir={artifacts.run_dir}"
    )
    _append_jsonl(
        Path(artifacts.run_dir) / "session" / "events.jsonl",
        {
            "ts": int(time.time()),
            "event": "pdf_processed",
            "pdf_name": artifacts.doc_name,
            "pdf_path": str(artifacts.pdf_path),
            "doc_id": artifacts.doc_id,
            "num_pages": artifacts.num_pages,
        },
    )
    return _join_logs(logs), artifacts.to_dict()


def ask_query(
    query: str,
    artifacts_state: Optional[Dict[str, str]],
) -> Tuple[str, str, Dict[str, object], List[Dict[str, object]]]:
    if not artifacts_state:
        return "Please process a PDF first.", "", {}, []
    if not query or not query.strip():
        return "Please enter a query.", "", {}, []

    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    artifacts = PipelineArtifacts.from_dict(artifacts_state)
    try:
        result = run_query(query.strip(), artifacts, logger=log)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Query failed: {exc}")
        return _join_logs(logs), "", {}, []

    final = result.get("final", {}) if isinstance(result, dict) else {}
    trace = result.get("trace", []) if isinstance(result, dict) else []
    final_text = ""
    if isinstance(final, dict):
        final_text = str(final.get("final_text") or "")
    _append_jsonl(
        Path(artifacts.run_dir) / "session" / "events.jsonl",
        {
            "ts": int(time.time()),
            "event": "query_run",
            "query": query.strip(),
            "final": final,
            "trace": trace,
        },
    )
    return _join_logs(logs), final_text, final, trace


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DocAgent Visualize") as demo:
        gr.Markdown("# DocAgent Visualize")
        gr.Markdown(
            "Upload a PDF, build indexes, then ask questions. "
            "Planner/executor traces are shown for inspection."
        )

        with gr.Row():
            pdf_file = gr.File(label="PDF File", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF")

        status = gr.Textbox(label="Status / Logs", lines=10, interactive=False)
        artifacts_state = gr.State()

        gr.Markdown("## Ask a question")
        query = gr.Textbox(label="Query", placeholder="Ask about the document...")
        ask_btn = gr.Button("Run Query")

        answer = gr.Markdown()
        final_json = gr.JSON(label="Final Result")
        trace_json = gr.JSON(label="Planner/Executor Trace")

        process_btn.click(
            process_pdf,
            inputs=[pdf_file],
            outputs=[status, artifacts_state],
        )

        ask_btn.click(
            ask_query,
            inputs=[query, artifacts_state],
            outputs=[status, answer, final_json, trace_json],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
