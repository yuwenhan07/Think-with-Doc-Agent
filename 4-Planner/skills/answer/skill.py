from __future__ import annotations

import json
from typing import Any, Dict

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json


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
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )

    content = completion.choices[0].message.content or ""
    parsed = safe_json(content)
    if not parsed:
        raise ValueError(f"answer: invalid JSON output: {content}")
    return parsed
