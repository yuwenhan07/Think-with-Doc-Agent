from __future__ import annotations

import json
from typing import Any, Dict

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    context = args.get("context", {})
    answer = args.get("answer", {})

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
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )

    content = completion.choices[0].message.content or ""
    parsed = safe_json(content)
    if not parsed:
        raise ValueError(f"judge_answer: invalid JSON output: {content}")
    return parsed
