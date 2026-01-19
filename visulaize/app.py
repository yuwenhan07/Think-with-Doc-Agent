from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from pipeline import PipelineArtifacts, build_pipeline, run_query


def _join_logs(lines: List[str]) -> str:
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _extract_retrieval(
    trace: List[Dict[str, object]],
) -> Tuple[List[Tuple[str, str]], str]:
    search_obs: Optional[Dict[str, object]] = None
    for item in reversed(trace):
        if item.get("tool") == "search":
            search_obs = item.get("observation")  # type: ignore[assignment]
            break

    if not isinstance(search_obs, dict):
        return [], "No retrieval results."

    block_hits = list(search_obs.get("block_hits", []))
    summary_hits = list(search_obs.get("summary_hits", []))

    images: List[Tuple[str, str]] = []
    text_lines: List[str] = []

    for hit in block_hits:
        if not isinstance(hit, dict):
            continue
        score = float(hit.get("score") or 0.0)
        page = hit.get("page_number")
        htype = hit.get("type") or "block"
        text = (hit.get("text") or "").strip()

        if htype != "text":
            asset_path = hit.get("asset_path")
            if asset_path:
                caption = f"p{page} {htype} {score:.3f}"
                if text:
                    caption += f" | {_truncate(text, 80)}"
                images.append((str(asset_path), caption))

        if text:
            text_lines.append(
                f"- [block {htype}] p{page} score {score:.3f}: {_truncate(text, 300)}"
            )

    for hit in summary_hits:
        if not isinstance(hit, dict):
            continue
        score = float(hit.get("score") or 0.0)
        page = hit.get("page_number")
        summary = (hit.get("summary") or "").strip()
        if summary:
            text_lines.append(
                f"- [summary] p{page} score {score:.3f}: {_truncate(summary, 300)}"
            )

    text_md = "\n".join(text_lines[:30]) if text_lines else "No text hits."
    return images, text_md


def _extract_planner_executor_io(
    trace: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    planner_io: List[Dict[str, object]] = []
    executor_io: List[Dict[str, object]] = []
    for item in trace:
        tool = item.get("tool")
        if tool == "planner":
            args = item.get("args", {})
            obs = item.get("observation", {})
            planner_io.append({
                "turn": args.get("turn") if isinstance(args, dict) else None,
                "input": args.get("input") if isinstance(args, dict) else None,
                "output": obs.get("output") if isinstance(obs, dict) else None,
                "raw": obs.get("raw") if isinstance(obs, dict) else None,
                "parse_error": obs.get("parse_error") if isinstance(obs, dict) else None,
                "fallback": obs.get("fallback") if isinstance(obs, dict) else None,
            })
            continue
        if tool == "planner_error":
            args = item.get("args", {})
            obs = item.get("observation", {})
            planner_io.append({
                "turn": args.get("turn") if isinstance(args, dict) else None,
                "input": args.get("input") if isinstance(args, dict) else None,
                "error": obs.get("error") if isinstance(obs, dict) else None,
            })
            continue
        executor_io.append({
            "tool": tool,
            "args": item.get("args"),
            "observation": item.get("observation"),
        })
    return planner_io, executor_io


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _memory_to_chat(memory: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for item in memory:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user") or "")
        assistant = str(item.get("assistant") or "")
        if user or assistant:
            rows.append((user, assistant))
    return rows


def process_pdf(pdf_file) -> Tuple[str, Optional[Dict[str, str]], List[Dict[str, object]], List[Tuple[str, str]]]:
    if pdf_file is None:
        return "Please upload a PDF file.", None, [], []

    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    try:
        artifacts = build_pipeline(pdf_file.name, logger=log)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Pipeline failed: {exc}")
        return _join_logs(logs), None, [], []

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
    return _join_logs(logs), artifacts.to_dict(), [], []


def ask_query(
    query: str,
    artifacts_state: Optional[Dict[str, str]],
    memory_state: List[Dict[str, object]],
) -> Tuple[
    str,
    str,
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Tuple[str, str]],
    str,
    List[Dict[str, object]],
    List[Tuple[str, str]],
]:
    if not artifacts_state:
        return "Please process a PDF first.", "", {}, [], [], [], [], "No retrieval results.", memory_state, _memory_to_chat(memory_state)
    if not query or not query.strip():
        return "Please enter a query.", "", {}, [], [], [], [], "No retrieval results.", memory_state, _memory_to_chat(memory_state)

    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    artifacts = PipelineArtifacts.from_dict(artifacts_state)
    try:
        result = run_query(query.strip(), artifacts, memory=memory_state, logger=log)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Query failed: {exc}")
        return _join_logs(logs), "", {}, [], [], [], [], "No retrieval results.", memory_state, _memory_to_chat(memory_state)

    final = result.get("final", {}) if isinstance(result, dict) else {}
    trace = result.get("trace", []) if isinstance(result, dict) else []
    final_text = ""
    if isinstance(final, dict):
        final_text = str(final.get("final_text") or "")
    planner_io, executor_io = _extract_planner_executor_io(trace if isinstance(trace, list) else [])
    retrieval_images, retrieval_text = _extract_retrieval(trace if isinstance(trace, list) else [])
    memory_state = list(memory_state)
    memory_state.append({"user": query.strip(), "assistant": final_text, "final": final})
    chat_history = _memory_to_chat(memory_state)
    _append_jsonl(
        Path(artifacts.run_dir) / "session" / "events.jsonl",
        {
            "ts": int(time.time()),
            "event": "query_run",
            "query": query.strip(),
            "final": final,
            "trace": trace,
            "memory": memory_state,
        },
    )
    return (
        _join_logs(logs),
        final_text,
        final,
        trace,
        planner_io,
        executor_io,
        retrieval_images,
        retrieval_text,
        memory_state,
        chat_history,
    )


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
        memory_state = gr.State([])

        gr.Markdown("## Ask a question")
        query = gr.Textbox(label="Query", placeholder="Ask about the document...")
        ask_btn = gr.Button("Run Query")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation")
                answer = gr.Markdown()
                final_json = gr.JSON(label="Final Result")
                trace_json = gr.JSON(label="Planner/Executor Trace")
            with gr.Column(scale=2):
                gr.Markdown("## Planner/Executor IO")
                planner_io = gr.JSON(label="Planner IO (Raw)")
                executor_io = gr.JSON(label="Executor IO (Raw)")
                gr.Markdown("## Retrieval Results")
                retrieval_images = gr.Gallery(
                    label="Image Hits",
                    columns=2,
                    height=360,
                    show_label=True,
                )
                retrieval_text = gr.Markdown()

        process_btn.click(
            process_pdf,
            inputs=[pdf_file],
            outputs=[status, artifacts_state, memory_state, chatbot],
        )

        ask_btn.click(
            ask_query,
            inputs=[query, artifacts_state, memory_state],
            outputs=[
                status,
                answer,
                final_json,
                trace_json,
                planner_io,
                executor_io,
                retrieval_images,
                retrieval_text,
                memory_state,
                chatbot,
            ],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
