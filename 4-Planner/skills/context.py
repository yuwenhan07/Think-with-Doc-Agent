from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


BASE_URL = "http://localhost:8003/v1"
BASE_MODEL = "Qwen3-VL-32B-Instruct"


@dataclass
class LLMConfig:
    api_key_env: str = "QianFan_API_KEY"
    base_url: str = BASE_URL
    model_id: str = BASE_MODEL
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
    locator: Dict[str, Any]
