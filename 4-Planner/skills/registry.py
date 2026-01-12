from __future__ import annotations

from typing import Callable, Dict

from .context import ExecutionContext, LLMConfig
from .rewrite.skill import execute as rewrite_execute
from .search.skill import execute as search_execute
from .judge_retrieval.skill import execute as judge_retrieval_execute
from .build_context.skill import execute as build_context_execute
from .answer.skill import execute as answer_execute
from .judge_answer.skill import execute as judge_answer_execute
from .finalize.skill import execute as finalize_execute

SkillFn = Callable[[dict, ExecutionContext, LLMConfig], dict]

SKILL_REGISTRY: Dict[str, SkillFn] = {
    "rewrite": rewrite_execute,
    "search": search_execute,
    "judge_retrieval": judge_retrieval_execute,
    "build_context": build_context_execute,
    "answer": answer_execute,
    "judge_answer": judge_answer_execute,
    "finalize": finalize_execute,
}


def get_skill(name: str) -> SkillFn:
    if name not in SKILL_REGISTRY:
        raise KeyError(f"Unknown skill: {name}")
    return SKILL_REGISTRY[name]
