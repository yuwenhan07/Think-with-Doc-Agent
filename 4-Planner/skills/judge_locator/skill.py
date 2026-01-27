from __future__ import annotations

import json
from typing import Any, Dict

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    locator_result = args.get("locator_result", {})

    prompt = (
        "You are the judge_locator skill. Output JSON only.\n"
        "Assess whether locator output is accurate and complete for the query.\n"
        "Return JSON with keys: verdict (good|bad|uncertain), state (complete|partial|missing).\n"
        "Do not include any other keys.\n"
        "If query specifies page/figure/table, evidence should include matching page and matching figure/table when possible.\n"
        f"Query: {query}\n"
        "Locator result (JSON):\n"
        f"{json.dumps(locator_result, ensure_ascii=False)}\n"
    )

    client = get_llm_client(llm)
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        response_format={
            "type": "json_object",
        },
    )

    content = completion.choices[0].message.content or ""
    parsed = safe_json(content)
    if not parsed:
        raise ValueError(f"judge_locator: invalid JSON output: {content}")
    verdict = parsed.get("verdict") if isinstance(parsed, dict) else None
    state = parsed.get("state") if isinstance(parsed, dict) else None
    if not verdict:
        verdict = "uncertain"
    if not state:
        state = "partial"
    return {"verdict": verdict, "state": state}
