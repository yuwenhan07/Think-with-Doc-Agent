"""Prompt builders for VLM/LLM understanding stage.

Keep all prompt text in this module so that other stages can reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class SummaryPromptConfig:
    """Configuration for page-level summarization prompts."""

    language: str = "en"  # "zh" or "en"
    max_bullets: int = 6
    max_sentences: int = 5


_TABULAR_BEGIN_RE = re.compile(r"\\begin\{tabular\}")
_TABULAR_END_RE = re.compile(r"\\end\{tabular\}")


def strip_long_tabular_blocks(text: str, *, keep_head_lines: int = 0) -> str:
    """Remove LaTeX tabular blocks to reduce token usage in summaries.

    Args:
        text: OCR markdown text.
        keep_head_lines: If > 0, keep the first N lines inside each tabular block.

    Returns:
        Cleaned text.
    """
    if not text:
        return text

    lines = text.splitlines()
    out: list[str] = []
    in_tabular = False
    kept = 0

    for line in lines:
        if not in_tabular and _TABULAR_BEGIN_RE.search(line):
            in_tabular = True
            kept = 0
            if keep_head_lines > 0:
                out.append(line)
            else:
                out.append("[TABULAR_BLOCK_REMOVED]")
            continue

        if in_tabular:
            if _TABULAR_END_RE.search(line):
                in_tabular = False
                if keep_head_lines > 0:
                    out.append(line)
                continue
            if keep_head_lines > 0 and kept < keep_head_lines:
                out.append(line)
                kept += 1
            continue

        out.append(line)

    return "\n".join(out)


def build_page_summary_prompt(
    text_raw: str,
    *,
    page_number: Optional[int] = None,
    config: SummaryPromptConfig = SummaryPromptConfig(),
) -> str:
    """Build a page-level summarization prompt.

    The model should produce a concise, citation-friendly summary of one page.
    """
    header = f"Page {page_number}" if page_number is not None else "This page"

    if config.language.lower() == "en":
        return (
            f"You are summarizing a single page of an academic PDF.\n"
            f"Summarize {header} using at most {config.max_sentences} sentences OR up to "
            f"{config.max_bullets} bullet points.\n\n"
            "Requirements:\n"
            "- Focus on the *main* ideas, definitions, claims, and any figure/table takeaway.\n"
            "- If the page is mostly references/appendix boilerplate, say so succinctly.\n"
            "- Do not invent content not present in the page.\n"
            "- Be specific (mention section/figure/table numbers if present).\n\n"
            "OCR markdown for the page:\n"
            "---\n"
            f"{text_raw}\n"
            "---\n"
        )

    # Default: Chinese
    return (
        "你在总结一篇学术 PDF 的单页内容。\n"
        f"请对{header}做一个简洁摘要：不超过{config.max_sentences}句，或不超过{config.max_bullets}个要点。\n\n"
        "要求：\n"
        "- 只总结页面中出现的主要观点/定义/方法/结论，以及图表要点（如有）。\n"
        "- 如果该页主要是参考文献/附录/模板性内容，请明确说明。\n"
        "- 不要编造页面中不存在的信息。\n"
        "- 尽量具体：如出现 Section/Figure/Table 编号，请点出。\n\n"
        "该页 OCR markdown 内容如下：\n"
        "---\n"
        f"{text_raw}\n"
        "---\n"
    )
