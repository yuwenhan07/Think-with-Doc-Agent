# DocAgent Visualize

This folder provides a minimal UI that lets you upload a PDF, run the full
DocAgent pipeline (render -> OCR -> summary -> blocks -> embeddings), and then
query the planner/executor while viewing trace output.

## Prerequisites
- Python 3.9+
- A valid `QianFan_API_KEY` in your environment (OCR, summaries, embeddings, and
  planner/executor all depend on it).
- `qwen_vl_utils` must be available in your environment (used by the OCR step).

## Install
```bash
pip install -r visulaize/requirements.txt
```

## Run
```bash
python visulaize/app.py
```

## Notes
- Output artifacts are stored under `visulaize/runs/`.
- Large PDFs can take a while depending on OCR and embedding latency.
