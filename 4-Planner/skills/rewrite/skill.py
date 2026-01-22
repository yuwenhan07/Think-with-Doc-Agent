from __future__ import annotations

from typing import Any, Dict

from ..context import ExecutionContext, LLMConfig
from ..llm_utils import get_llm_client, safe_json


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    query = args.get("query", "")
    max_rewrites = int(args.get("max_rewrites", 3))
    memory = args.get("memory") or ""

    prompt = (
        "You are the rewrite skill. Output JSON only.\n"
        "Your task is to provide a set of rewrites for a given question based on the context provided."
        "1. Identify the essential problem."
        "2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail."
        "3. Draft an answer with as many thoughts as you have"
        f"Memory: {memory}\n"
        f"Query: {query}\n"
        f"Max rewrites: {max_rewrites}\n"
    )

    client = get_llm_client(llm)
    completion = client.chat.completions.create(
        model=llm.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        response_format={
            'type': 'json_object'
        },
    )

    content = completion.choices[0].message.content or ""
    parsed = safe_json(content)
    if not parsed:
        raise ValueError(f"rewrite: invalid JSON output: {content}")

    rewrites = parsed.get("rewrites", [])
    if isinstance(rewrites, list) and len(rewrites) > max_rewrites:
        parsed["rewrites"] = rewrites[:max_rewrites]

    return parsed
