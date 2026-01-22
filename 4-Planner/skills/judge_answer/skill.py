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
    query = args.get("query", "")
    context = args.get("context", {})
    answer = args.get("answer", {})
    evidence = context.get("evidence", []) if isinstance(context.get("evidence", []), list) else []

    prompt = (
        "You are the judge_answer skill. Output JSON only.\n"
        "Check evidence consistency and citation coverage.\n"
        "Return: verdict (final|revise|need_more_evidence), issues, next_actions.\n"
        f"Query: {query}\n"
        "Context (JSON):\n"
        f"{json.dumps(context, ensure_ascii=False)}\n"
        "Answer (JSON):\n"
        f"{json.dumps(answer, ensure_ascii=False)}\n"
    )

    client = get_llm_client(llm)
    messages = _build_multimodal_messages(prompt, evidence)
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=messages,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        response_format={
            'type': 'json_object'
        },
    )

    content = completion.choices[0].message.content or ""
    parsed = safe_json(content)
    if not parsed:
        raise ValueError(f"judge_answer: invalid JSON output: {content}")
    return parsed
