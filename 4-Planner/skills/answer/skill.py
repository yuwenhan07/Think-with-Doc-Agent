from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json


def _encode_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _build_multimodal_messages(prompt: str, evidence: List[Dict[str, Any]], *, max_images: int = 4) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    used = 0
    for item in evidence:
        asset_path = item.get("asset_path")
        if not asset_path:
            continue
        p = Path(str(asset_path))
        if not p.exists():
            continue
        if used >= max_images:
            break
        b64 = _encode_image(str(p))
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        used += 1
    return [{"role": "user", "content": contents}]


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    context = args.get("context", {})
    if not isinstance(context, dict):
        context = {}
    need_citations = bool(args.get("need_citations", True))
    style = args.get("style", "short")

    evidence = context.get("evidence", []) if isinstance(context.get("evidence", []), list) else []
    if not evidence:
        return {
            "answer": "not answerable",
            "citations": [],
            "confidence": 0.0,
        }

    prompt = (
        "You are the answer skill. Use ONLY the provided evidence.\n"
        "Output JSON only with fields: answer, citations, confidence.\n"
        "Citations must reference page + block_id and include a short quote.\n"
        f"Need citations: {need_citations}\n"
        f"Style: {style}\n"
        f"Question: {context.get('question', '')}\n"
        "Evidence:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}\n"
    )

    client = get_llm_client(llm)
    messages = _build_multimodal_messages(prompt, evidence)
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=messages,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )

    content = completion.choices[0].message.content or ""
    parsed = safe_json(content)
    if not parsed:
        raise ValueError(f"answer: invalid JSON output: {content}")
    return parsed
