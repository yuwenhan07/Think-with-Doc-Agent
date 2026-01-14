from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from .context import LLMConfig


def get_llm_client(cfg: LLMConfig) -> OpenAI:
    api_key = (
        os.environ.get(cfg.api_key_env)
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    return OpenAI(base_url=cfg.base_url, api_key=api_key)


def safe_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
