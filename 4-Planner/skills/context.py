from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LLMConfig:
    api_key_env: str = "QianFan_API_KEY"
    base_url: str = "http://localhost:8003/v1"
    model_id: str = "models/Qwen3-VL-32B-Instruct"
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
