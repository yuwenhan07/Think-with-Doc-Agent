from __future__ import annotations

import importlib.util
import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss

from planners import LLMPlanner, PlannerConfig
from skills import ExecutionContext, LLMConfig, get_skill


@dataclass
class BudgetConfig:
    max_turns: int = 8
    max_search_calls: int = 3
    max_rewrite_calls: int = 2
    max_blocks_context: int = 8
    max_same_query_search: int = 2


@dataclass
class ExecutionState:
    query: str
    turn: int = 0
    last_tool: Optional[str] = None
    last_observation: Optional[Dict[str, Any]] = None
    last_search_result: Optional[Dict[str, Any]] = None
    last_context: Optional[Dict[str, Any]] = None
    last_answer: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    search_calls: int = 0
    rewrite_calls: int = 0
    answer_calls: int = 0
    search_query_counts: Dict[str, int] = field(default_factory=dict)


def _load_search_module(root: Path):
    search_path = root / "3-embedding" / "search.py"
    spec = importlib.util.spec_from_file_location("search_module", search_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load search module at {search_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Executor:
    def __init__(
        self,
        *,
        index_dir: Path,
        asset_base_dir: Path,
        planner_config: Optional[PlannerConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        budget: Optional[BudgetConfig] = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        search_module = _load_search_module(root)

        summary_index = faiss.read_index(str(index_dir / "summary.index.faiss"))
        summary_meta = search_module.load_jsonl(str(index_dir / "summary.meta.jsonl"))
        manifest_path = index_dir / "blocks.manifest.json"
        manifest = search_module.load_json(str(manifest_path)) if manifest_path.exists() else {"pages": []}

        self.ctx = ExecutionContext(
            index_dir=index_dir,
            asset_base_dir=asset_base_dir,
            search_module=search_module,
            summary_index=summary_index,
            summary_meta=summary_meta,
            manifest=manifest,
        )

        self.planner = LLMPlanner(planner_config)
        self.llm_config = llm_config or LLMConfig()
        self.budget = budget or BudgetConfig()

    def _budget_snapshot(self, state: ExecutionState) -> Dict[str, Any]:
        return {
            "turns": f"{state.turn}/{self.budget.max_turns}",
            "search_calls": f"{state.search_calls}/{self.budget.max_search_calls}",
            "rewrite_calls": f"{state.rewrite_calls}/{self.budget.max_rewrite_calls}",
            "max_blocks_context": self.budget.max_blocks_context,
        }

    def _summarize_history(self, state: ExecutionState, limit: int = 8) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return state.history[-limit:]

    def _summarize_observation(self, state: ExecutionState) -> Dict[str, Any]:
        if not state.last_observation or not state.last_tool:
            return {}
        obs = state.last_observation
        if state.last_tool == "search":
            stats = obs.get("stats", {}) if isinstance(obs, dict) else {}
            return {
                "tool": "search",
                "stats": {
                    "block_hits": stats.get("block_hits"),
                    "has_abstract": stats.get("has_abstract"),
                },
                "candidate_pages": obs.get("candidate_pages") if isinstance(obs, dict) else None,
            }
        if state.last_tool == "build_context":
            context = obs.get("context", {}) if isinstance(obs, dict) else {}
            evidence = context.get("evidence", []) if isinstance(context, dict) else []
            pages = []
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                page = item.get("page")
                if page is not None:
                    pages.append(page)
            pages = sorted(set(pages))
            return {
                "tool": "build_context",
                "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
                "pages": pages[:5],
            }
        if state.last_tool == "judge_retrieval":
            return {
                "tool": "judge_retrieval",
                "verdict": obs.get("verdict") if isinstance(obs, dict) else None,
                "suggestions": obs.get("suggestions") if isinstance(obs, dict) else None,
            }
        if state.last_tool == "answer":
            return {
                "tool": "answer",
                "answer_preview": (obs.get("answer") if isinstance(obs, dict) else None),
                "confidence": obs.get("confidence") if isinstance(obs, dict) else None,
            }
        if state.last_tool == "judge_answer":
            return {
                "tool": "judge_answer",
                "verdict": obs.get("verdict") if isinstance(obs, dict) else None,
                "issues": obs.get("issues") if isinstance(obs, dict) else None,
            }
        return {"tool": state.last_tool}

    def _record(self, state: ExecutionState, tool: str, args: Dict[str, Any], obs: Dict[str, Any]) -> None:
        state.history.append({"tool": tool, "summary": {k: obs.get(k) for k in ("verdict", "stats", "issues") if k in obs}})
        state.trace.append({"tool": tool, "args": args, "observation": obs})
        state.last_tool = tool
        state.last_observation = obs

    def _record_planner(
        self,
        state: ExecutionState,
        plan_input: Dict[str, Any],
        plan_output: Optional[Dict[str, Any]],
        trace: Dict[str, Any],
    ) -> None:
        state.history.append({"tool": "planner", "summary": {"parse_error": trace.get("parse_error")}})
        state.trace.append({
            "tool": "planner",
            "args": {"turn": state.turn, "input": plan_input},
            "observation": {
                "output": plan_output,
                "raw": trace.get("raw"),
                "parse_error": trace.get("parse_error"),
                "fallback": trace.get("fallback"),
            },
        })

    def _force_judge(self, state: ExecutionState) -> Optional[Dict[str, Any]]:
        if state.last_tool == "search":
            return {
                "tool": "judge_retrieval",
                "args": {"query": state.query, "search_result": state.last_search_result},
            }
        if state.last_tool == "answer":
            return {
                "tool": "judge_answer",
                "args": {"query": state.query, "context": state.last_context, "answer": state.last_answer},
            }
        return None

    def _fallback_plan(self, state: ExecutionState, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "final" in plan:
            if state.last_search_result and not state.last_context:
                return {"tool": "build_context", "args": {"max_blocks": self.budget.max_blocks_context}}
            if state.last_context and not state.last_answer:
                return {"tool": "answer", "args": {"need_citations": True, "style": "short"}}
            if state.last_answer and state.last_tool != "judge_answer":
                return {
                    "tool": "judge_answer",
                    "args": {"query": state.query, "context": state.last_context, "answer": state.last_answer},
                }
        return None

    def _apply_budget(self, state: ExecutionState, tool: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if tool == "search":
            if state.search_calls >= self.budget.max_search_calls:
                return {"final_text": "Search budget exceeded.", "citations": []}
            q = args.get("query") or state.query
            state.search_query_counts[q] = state.search_query_counts.get(q, 0) + 1
            if state.search_query_counts[q] > self.budget.max_same_query_search:
                return {"final_text": "Repeated search exceeded limit.", "citations": []}
            state.search_calls += 1
        if tool == "rewrite":
            if state.rewrite_calls >= self.budget.max_rewrite_calls:
                return {"final_text": "Rewrite budget exceeded.", "citations": []}
            state.rewrite_calls += 1
        if tool == "answer":
            state.answer_calls += 1
        return None

    def _normalize_args(self, state: ExecutionState, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Inject stateful defaults to avoid planner omission causing empty context.
        if tool == "build_context":
            if "search_result" not in args:
                args["search_result"] = state.last_search_result or {}
            if "query" not in args:
                args["query"] = state.query
        elif tool == "answer":
            if "answer" in args and "context" not in args:
                # Planner may output a mistaken schema; ignore and recover.
                args.pop("answer", None)
            if "context" not in args:
                args["context"] = state.last_context or {}
        elif tool == "judge_answer":
            if "context" not in args:
                args["context"] = state.last_context or {}
            if "answer" not in args:
                args["answer"] = state.last_answer or {}
        elif tool == "judge_retrieval":
            if "search_result" not in args:
                args["search_result"] = state.last_search_result or {}
            if "query" not in args:
                args["query"] = state.query
        return args

    def _final_not_answerable(self) -> Dict[str, Any]:
        return {"final_text": "not answerable", "citations": [], "confidence": 0.0}

    def _final_from_state(self, state: ExecutionState) -> Dict[str, Any]:
        if state.last_answer:
            finalize = get_skill("finalize")
            return finalize({"answer": state.last_answer}, self.ctx, self.llm_config)
        if state.last_context:
            answer = get_skill("answer")
            ans = answer({"context": state.last_context, "need_citations": True, "style": "short"}, self.ctx, self.llm_config)
            finalize = get_skill("finalize")
            return finalize({"answer": ans}, self.ctx, self.llm_config)
        return self._final_not_answerable()

    def _finalize_with_judge(self, state: ExecutionState) -> Optional[Dict[str, Any]]:
        if not state.last_context and state.last_search_result:
            pruned = self._prune_search_result(state.last_search_result)
            ctx_obs = get_skill("build_context")(
                {"search_result": pruned, "max_blocks": self.budget.max_blocks_context, "query": state.query},
                self.ctx,
                self.llm_config,
            )
            state.last_context = ctx_obs.get("context")
            self._record(state, "build_context", {"max_blocks": self.budget.max_blocks_context}, ctx_obs)

        if not state.last_context:
            return None

        if not state.last_answer:
            ans_obs = get_skill("answer")(
                {"context": state.last_context, "need_citations": True, "style": "short"},
                self.ctx,
                self.llm_config,
            )
            state.last_answer = ans_obs
            self._record(state, "answer", {"style": "short"}, ans_obs)

        judge_obs = get_skill("judge_answer")(
            {"query": state.query, "context": state.last_context, "answer": state.last_answer},
            self.ctx,
            self.llm_config,
        )
        self._record(state, "judge_answer", {}, judge_obs)
        if judge_obs.get("verdict") == "final":
            final_obs = get_skill("finalize")({"answer": state.last_answer}, self.ctx, self.llm_config)
            self._record(state, "finalize", {}, final_obs)
            return final_obs
        return None

    def _prune_search_result(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        if not search_result:
            return {}
        block_hits = list(search_result.get("block_hits", []))
        block_hits.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
        keep_blocks = block_hits[: max(self.budget.max_blocks_context * 2, 8)]

        summary_hits = list(search_result.get("summary_hits", []))
        summary_hits.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
        keep_summaries = summary_hits[:5]

        candidate_pages = sorted({
            b.get("page_number") for b in keep_blocks + keep_summaries if b.get("page_number") is not None
        })

        pruned = dict(search_result)
        pruned["block_hits"] = keep_blocks
        pruned["summary_hits"] = keep_summaries
        pruned["candidate_pages"] = candidate_pages
        return pruned

    def run(self, query: str) -> Dict[str, Any]:
        state = ExecutionState(query=query)

        while state.turn < self.budget.max_turns:
            state.turn += 1
            forced = self._force_judge(state)
            if forced:
                plan = forced
            else:
                planner_input = {
                    "query": state.query,
                    "turn": state.turn,
                    "budget": self._budget_snapshot(state),
                    "history": self._summarize_history(state),
                    "last_observation": self._summarize_observation(state),
                }
                plan = None
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        plan, planner_trace = self.planner.plan_with_trace(planner_input)
                        self._record_planner(state, planner_input, copy.deepcopy(plan), planner_trace)
                        if plan:
                            break
                    except Exception as exc:
                        state.trace.append({
                            "tool": "planner_error",
                            "args": {"turn": state.turn, "input": planner_input, "attempt": attempt + 1},
                            "observation": {"error": str(exc)},
                        })
                        plan = None
                    if attempt == max_retries:
                        return {"final": {"final_text": "Planner output is not valid JSON.", "citations": []}, "trace": state.trace}
                if not plan:
                    fallback = self._fallback_plan(state, {"final": {}})
                    if fallback:
                        plan = fallback
                    else:
                        return {"final": {"final_text": "Planner output is not valid JSON.", "citations": []}, "trace": state.trace}

            fallback = self._fallback_plan(state, plan)
            if fallback:
                plan = fallback

            if "final" in plan:
                final_obj = plan.get("final")
                if isinstance(final_obj, dict) and final_obj:
                    return {"final": final_obj, "trace": state.trace}
                return {"final": self._final_from_state(state), "trace": state.trace}

            tool = plan.get("tool")
            args = plan.get("args", {})
            if tool is None:
                planner_input = {
                    "query": state.query,
                    "turn": state.turn,
                    "budget": self._budget_snapshot(state),
                    "history": self._summarize_history(state),
                    "last_observation": self._summarize_observation(state),
                }
                plan = None
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        plan, planner_trace = self.planner.plan_with_trace(planner_input)
                        self._record_planner(state, planner_input, copy.deepcopy(plan), planner_trace)
                        tool = plan.get("tool") if plan else None
                        args = plan.get("args", {}) if plan else {}
                        if tool:
                            break
                    except Exception as exc:
                        state.trace.append({
                            "tool": "planner_error",
                            "args": {"turn": state.turn, "input": planner_input, "attempt": attempt + 1},
                            "observation": {"error": str(exc)},
                        })
                        plan = None
                    if attempt == max_retries:
                        return {"final": {"final_text": "Planner returned no tool.", "citations": []}, "trace": state.trace}
                if tool is None:
                    return {"final": {"final_text": "Planner returned no tool.", "citations": []}, "trace": state.trace}

            budget_fail = self._apply_budget(state, tool, args)
            if budget_fail:
                final_obs = self._finalize_with_judge(state)
                if final_obs:
                    return {"final": final_obs, "trace": state.trace}
                return {"final": self._final_not_answerable(), "trace": state.trace}

            args = self._normalize_args(state, tool, args)
            if tool == "build_context" and not args.get("search_result"):
                return {"final": self._final_not_answerable(), "trace": state.trace}
            if tool == "answer" and not args.get("context"):
                return {"final": self._final_not_answerable(), "trace": state.trace}

            skill = get_skill(tool)
            try:
                obs = skill(args, self.ctx, self.llm_config)
            except Exception as exc:
                return {
                    "final": {"final_text": f"Skill failed: {tool}: {exc}", "citations": []},
                    "trace": state.trace,
                }

            if tool == "search":
                state.last_search_result = obs
            if tool == "build_context":
                state.last_context = obs.get("context")
            if tool == "answer":
                state.last_answer = obs
            if tool == "judge_answer":
                evidence = (state.last_context or {}).get("evidence", [])
                if not evidence and obs.get("verdict") == "final":
                    obs["verdict"] = "need_more_evidence"
                    obs["issues"] = list(dict.fromkeys((obs.get("issues") or []) + ["empty_evidence"]))
                    obs["next_actions"] = [
                        {
                            "action": "search",
                            "query_hint": "abstract introduction contributions",
                            "filters": {"avoid_sections": ["references", "bibliography"]},
                        }
                    ]

            self._record(state, tool, args, obs)

            if tool == "judge_answer" and obs.get("verdict") == "final":
                final_obs = get_skill("finalize")({"answer": state.last_answer or {}}, self.ctx, self.llm_config)
                self._record(state, "finalize", {}, final_obs)
                return {"final": final_obs, "trace": state.trace}

            if tool == "finalize":
                return {"final": obs, "trace": state.trace}

        if state.last_context:
            ans_obs = get_skill("answer")(
                {"context": state.last_context, "need_citations": True, "style": "short"},
                self.ctx,
                self.llm_config,
            )
            state.last_answer = ans_obs
            self._record(state, "answer", {"style": "short"}, ans_obs)

            judge_obs = get_skill("judge_answer")(
                {"query": state.query, "context": state.last_context, "answer": state.last_answer},
                self.ctx,
                self.llm_config,
            )
            self._record(state, "judge_answer", {}, judge_obs)
            if judge_obs.get("verdict") == "final":
                final_obs = get_skill("finalize")({"answer": state.last_answer or {}}, self.ctx, self.llm_config)
                self._record(state, "finalize", {}, final_obs)
                return {"final": final_obs, "trace": state.trace}

        return {"final": self._final_not_answerable(), "trace": state.trace}
