from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LLMConfig:
    api_key_env: str = "QianFan_API_KEY"
    base_url: str = "https://qianfan.baidubce.com/v2"
    model_id: str = "qwen3-vl-235b-a22b-instruct"
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass
class ExecutionContext:
    index_dir: Path
    asset_base_dir: Path
    search_module: Any
    summary_index: Any
    summary_meta: List[Dict[str, Any]]
    manifest: Dict[str, Any]
