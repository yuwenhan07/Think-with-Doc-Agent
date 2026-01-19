from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


@dataclass
class PlannerConfig:
    api_key_env: str = "QianFan_API_KEY"
    base_url: str = "https://qianfan.baidubce.com/v2"
    model_id: str = "qwen3-vl-235b-a22b-instruct"
    temperature: float = 0.2
    max_tokens: int = 256


class LLMPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        self.config = config or PlannerConfig()
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {self.config.api_key_env}")
        self.client = OpenAI(base_url=self.config.base_url, api_key=api_key)

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_str = not in_str
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fragment = self._extract_balanced_json(text)
        if fragment:
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                pass
        return None

    def _tool_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\"tool\"\\s*:\\s*\"([a-zA-Z_]+)\"", text)
        if not m:
            return None
        return {"tool": m.group(1), "args": {}}

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        return (
            "You are a tool orchestrator. Output JSON only.\n"
            "Allowed outputs:\n"
            "  {\"tool\": \"rewrite\", \"args\": {...}}\n"
            "  {\"tool\": \"search\", \"args\": {...}}\n"
            # "  {\"tool\": \"judge_retrieval\", \"args\": {...}}\n"
            "  {\"tool\": \"build_context\", \"args\": {...}}\n"
            "  {\"tool\": \"answer\", \"args\": {...}}\n"
            # "  {\"tool\": \"judge_answer\", \"args\": {...}}\n"
            "  {\"tool\": \"finalize\", \"args\": {...}}\n"
            "  {\"final\": {...}}\n"
            "Tool purposes:\n"
            "- rewrite: improve recall by reformulating the query into better retrieval queries.\n"
            "- search: retrieve relevant pages/blocks from the original PDF.\n"
            # "- judge_retrieval: assess whether search results are sufficient/clean; suggest next action.\n"
            "- build_context: select a small, high-signal evidence set from search hits.\n"
            "- answer: generate the response from the built context.\n"
            # "- judge_answer: verify the answer quality and decide if more evidence is needed.\n"
            "- finalize: produce the final user-facing output.\n"
            "Constraints:\n"
            "- After answer, must call judge_answer.\n"
            "- After search, must call judge_retrieval.\n"
            "- Args must be minimal. Do NOT include full passages, blocks, or long text.\n"
            "- Use only small scalars and small lists (e.g., query, k_pages, filters).\n"
            f"Query: {state.get('query', '')}\n"
            f"Turn: {state.get('turn', 0)}\n"
            f"Budget: {state.get('budget', {})}\n"
            f"History: {state.get('history', [])}\n"
            f"Last observation: {state.get('last_observation', {})}\n"
            f"Memory: {state.get('memory', '')}\n"
        )

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan, _ = self.plan_with_trace(state)
        if not plan:
            raise ValueError("Planner output is not valid JSON.")
        return plan

    def plan_with_trace(self, state: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        prompt = self._build_prompt(state)
        completion = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = completion.choices[0].message.content or ""
        parsed = self._safe_json(content)
        trace: Dict[str, Any] = {
            "raw": content,
            "parse_error": None,
            "fallback": None,
        }
        if not parsed:
            parsed = self._tool_from_text(content)
            trace["fallback"] = "tool_from_text"
        if not parsed:
            trace["parse_error"] = "invalid_json"
        return parsed, trace
