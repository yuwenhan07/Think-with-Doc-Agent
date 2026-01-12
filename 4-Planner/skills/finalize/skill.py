from __future__ import annotations

from typing import Any, Dict

from ..context import ExecutionContext, LLMConfig


def execute(args: Dict[str, Any], ctx: ExecutionContext, llm: LLMConfig) -> Dict[str, Any]:
    answer = args.get("answer", {})
    citations = answer.get("citations", []) or []
    return {
        "final_text": answer.get("answer", ""),
        "citations": citations,
        "confidence": answer.get("confidence", 0.0),
    }
