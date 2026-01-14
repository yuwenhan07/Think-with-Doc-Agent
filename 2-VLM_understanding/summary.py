from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from openai import OpenAI

from prompt import (
    SummaryPromptConfig,
    build_page_summary_prompt,
    strip_long_tabular_blocks,
)


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class QianfanVLMClient:
    """Qianfan / Qwen-VL compatible multimodal client (image + text)."""

    def __init__(
        self,
        *,
        api_key_env: str = "QianFan_API_KEY",
        base_url: str = "http://localhost:8003/v1",
        model_id: str = "qwen3-vl-32B",
        min_pixels: int = 512 * 32 * 32,
        max_pixels: int = 2048 * 32 * 32,
    ) -> None:
        api_key = (
            os.environ.get(api_key_env)
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def chat_with_image(self, *, image_path: str, prompt: str) -> str:
        """Send one image + one text prompt and return model output."""
        base64_image = _encode_image_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        return completion.choices[0].message.content.strip()


def summarize_page_text(
    text_raw: str,
    *,
    page_image_path: str,
    page_number: Optional[int] = None,
    prompt_config: SummaryPromptConfig = SummaryPromptConfig(),
    strip_tabular: bool = True,
    keep_tabular_head_lines: int = 0,
    max_chars: int = 10240,
    temperature: float = 0.2,
) -> str:
    """Summarize a single page OCR markdown text.

    Args:
        text_raw: OCR markdown for the page.
        page_number: Optional page number for prompt context.
        page_image_path: Path to the page image file.
        prompt_config: Language and style constraints.
        strip_tabular: Remove long LaTeX tabular blocks to reduce token usage.
        keep_tabular_head_lines: If stripping is enabled, keep first N lines inside tabular.
        max_chars: Hard cap to avoid oversized requests.
        temperature: Sampling temperature.

    Returns:
        A concise summary string.
    """
    if text_raw is None:
        text_raw = ""

    cleaned = text_raw
    if strip_tabular:
        cleaned = strip_long_tabular_blocks(cleaned, keep_head_lines=keep_tabular_head_lines)

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n[TRUNCATED]"

    user_prompt = build_page_summary_prompt(
        cleaned,
        page_number=page_number,
        config=prompt_config,
    )
    # print(f"[DEBUG] Prompt:\n{user_prompt}")
    vlm_client = QianfanVLMClient()
    return vlm_client.chat_with_image(
        image_path=page_image_path,
        prompt=user_prompt,
    )


def add_page_summaries_inplace(
    doc: Dict[str, Any],
    *,
    text_key: str = "text_raw",
    pages_key: str = "pages",
    summary_key: str = "page_summary",
    prompt_config: SummaryPromptConfig = SummaryPromptConfig(),
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    """Generate per-page summaries and write them into doc[pages][i][summary_key]."""
    pages = doc.get(pages_key, [])
    for p in pages:
        page_number = p.get("page_number")
        text_raw = p.get(text_key, "")
        p[summary_key] = summarize_page_text(
            text_raw,
            page_image_path=p.get("image_path"),
            page_number=page_number,
            prompt_config=prompt_config,
        )
        if sleep_s > 0:
            time.sleep(sleep_s)

    return doc


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def summarize_doc_json(input_json_path: str, output_json_path: str) -> None:
    """Test demo: summarize each page in a single doc JSON and save."""
    doc = read_json(input_json_path)
    add_page_summaries_inplace(doc, prompt_config=SummaryPromptConfig(language="en"))
    write_json(output_json_path, doc)


def summarize_pages_jsonl(input_jsonl_path: str, output_jsonl_path: str) -> None:
    """Test demo: summarize each line (page object) in a page-level JSONL and save."""
    items = read_jsonl(input_jsonl_path)

    out: List[Dict[str, Any]] = []
    for page in items:
        page_number = page.get("page_number")
        text_raw = page.get("text_raw", "")
        page["page_summary"] = summarize_page_text(
            text_raw,
            page_image_path=page.get("image_path"),
            page_number=page_number,
            prompt_config=SummaryPromptConfig(language="en"),
        )
        out.append(page)

    write_jsonl(output_jsonl_path, out)
