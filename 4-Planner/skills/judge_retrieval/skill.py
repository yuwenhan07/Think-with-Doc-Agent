from __future__ import annotations

import json
from typing import Any, Dict

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    search_result = args.get("search_result", {})

    prompt = (
        "You are the judge_retrieval skill. Output JSON only.\n"
        "Assess if retrieved evidence is sufficient and clean.\n"
        "Return: verdict (good|bad|uncertain), reasons, suggestions.\n"
        "suggestions is a list of {action, ...} for rewrite/search.\n"
        f"Query: {query}\n"
        "Search result (JSON):\n"
        f"{json.dumps(search_result, ensure_ascii=False)}\n"
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
        raise ValueError(f"judge_retrieval: invalid JSON output: {content}")
    return parsed
