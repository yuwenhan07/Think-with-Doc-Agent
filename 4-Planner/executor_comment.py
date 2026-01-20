"""
executor.py
当成“状态机 + 预算控制 + 工具调度”的统一执行器。
"""
from __future__ import annotations

import importlib.util
import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss

# * Executor 依赖于 planners 和 skills 模块，导入对应的 planners 和 skills
from planners import LLMPlanner, PlannerConfig
from skills import ExecutionContext, LLMConfig, get_skill

# * 预算配置
@dataclass
class BudgetConfig:
    max_turns: int = 8   # * max_turns 最大对话轮数  无论是 judge 还是 planner 选工具，都算一轮，每进一次循环就加一次
    max_search_calls: int = 3  # * max_search_calls 最大搜索调用次数
    max_rewrite_calls: int = 2  # * max_rewrite_calls 最大重写调用次数
    max_blocks_context: int = 8  # * max_blocks_context 最大上下文块数
    max_same_query_search: int = 2 # * max_same_query_search 同一查询的最大搜索次数

# * 执行状态，用于跨turn传递信息，做预算控制，并且生成trace
@dataclass
class ExecutionState:
    query: str  # * query 当前query
    turn: int = 0  # * turn 当前turn
    # * 可传入 上一个工具、上一个观察结果、上一次搜索结果、上一次上下文、上一次答案
    last_tool: Optional[str] = None
    last_observation: Optional[Dict[str, Any]] = None 
    last_search_result: Optional[Dict[str, Any]] = None 
    last_context: Optional[Dict[str, Any]] = None
    last_answer: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list) # * history 历史记录
    trace: List[Dict[str, Any]] = field(default_factory=list)  # * trace 执行过程
    search_calls: int = 0
    rewrite_calls: int = 0
    answer_calls: int = 0
    search_query_counts: Dict[str, int] = field(default_factory=dict)


# * 手动import了search.py模块，以便在Executor中使用search相关功能
def _load_search_module(root: Path):
    search_path = root / "3-embedding" / "search.py"
    spec = importlib.util.spec_from_file_location("search_module", search_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load search module at {search_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Executor:
    # * Executor 初始化，加载索引和清单文件，创建执行上下文
    def __init__(
        self,
        *,
        index_dir: Path,
        asset_base_dir: Path,
        planner_config: Optional[PlannerConfig] = None, # * planner_config 规划器配置
        llm_config: Optional[LLMConfig] = None,  # * llm_config LLM配置
        budget: Optional[BudgetConfig] = None,  # * budget 预算配置
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        # * 加载search模块
        search_module = _load_search_module(root)

        # * 加载索引和清单文件
        summary_index = faiss.read_index(str(index_dir / "summary.index.faiss"))
        summary_meta = search_module.load_jsonl(str(index_dir / "summary.meta.jsonl"))
        manifest_path = index_dir / "blocks.manifest.json"
        manifest = search_module.load_json(str(manifest_path)) if manifest_path.exists() else {"pages": []}

        # * 创建执行上下文
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
        """返回当前预算使用情况的快照。"""
        return {
            "turns": f"{state.turn}/{self.budget.max_turns}",
            "search_calls": f"{state.search_calls}/{self.budget.max_search_calls}",
            "rewrite_calls": f"{state.rewrite_calls}/{self.budget.max_rewrite_calls}",
            "max_blocks_context": self.budget.max_blocks_context,
        }

    def _summarize_history(self, state: ExecutionState, limit: int = 8) -> List[Dict[str, Any]]:
        """返回最近的历史记录摘要。只取最近的N条历史，避免上下文太长。"""
        if limit <= 0:
            return []
        return state.history[-limit:]

    def _summarize_observation(self, state: ExecutionState) -> Dict[str, Any]:
        """返回最近一次观察结果的摘要。根据不同工具，提取关键信息。"""
        # * 根据上一个工具类型，提取不同的关键信息，生成摘要，如果没有则返回空字典。
        if not state.last_observation or not state.last_tool:
            return {}
        obs = state.last_observation
        # * 如果上一个工具是search、build_context、judge_retrieval、answer、judge_answer，则提取相应的关键信息
        if state.last_tool == "search": 
            stats = obs.get("stats", {}) if isinstance(obs, dict) else {}
            return {
                "tool": "search",
                "stats": {
                    "block_hits": stats.get("block_hits"),
                    "has_abstract": stats.get("has_abstract"),  # TODO: 这个字段没有用到，目前没有设置 has abstract
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
                # TODO: 这里现在取了10个页面，是否可以减少一点
                "pages": pages[:10],  # * 只取前10个页面
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
        """辅助函数，记录规划器的输入输出和跟踪信息。"""
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
        """如果上一个工具是 search 或着 answer，则强制调用对应的 judge 工具。 判断search或着answer的结果是否合格"""
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
        """planner 给了 final 但状态不完整时，自动补齐流程（比如：先 build_context，再 answer，再 judge）。"""
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
        """应用预算控制，返回是否超出预算的最终结果。如果超出预算，则返回最终结果，否则返回None。"""
        # TODO: 修改这一部分的预算控制，如果超过了预算，应该回到上一个合理的状态，而不是直接返回最终结果
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
        """如果没有正常结束，state中有答案或上下文，则抽取得到最终结果。"""
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
        """预算耗尽时尽量产出结果：补齐 context -> answer -> judge -> finalize。"""
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
        """搜索命中过多时剪枝，保留 top block、top summary，并生成候选页。"""
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

        # * 主循环
        while state.turn < self.budget.max_turns:
            state.turn += 1  # * 只要进入一次循环，就认为进行了一轮对话，轮数+1
            forced = self._force_judge(state)
            if forced:
                plan = forced
            else:
                # * 构造planner的输入
                planner_input = {
                    "query": state.query,
                    "turn": state.turn,
                    "budget": self._budget_snapshot(state),
                    "history": self._summarize_history(state),
                    "last_observation": self._summarize_observation(state),
                }
                # * planner重试：解析失败/异常时重试两次（plan为空/JSON解析失败）
                plan = None
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        plan, planner_trace = self.planner.plan_with_trace(planner_input)  # * 调用planner，返回plan和trace
                        self._record_planner(state, planner_input, copy.deepcopy(plan), planner_trace)
                        if plan:
                            break
                    # * 异常处理
                    except Exception as exc:
                        state.trace.append({
                            "tool": "planner_error",
                            "args": {"turn": state.turn, "input": planner_input, "attempt": attempt + 1},
                            "observation": {"error": str(exc)},
                        })
                        plan = None
                    if attempt == max_retries:
                        return {"final": {"final_text": "Planner output is not valid JSON.", "citations": []}, "trace": state.trace}
                if not plan:  # * 如果plan为空，则尝试使用fallback plan，错误兜底逻辑
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

            # * 直接从plan中提取tool和args
            tool = plan.get("tool")
            args = plan.get("args", {})
            if tool is None:
                # TODO：这里应该有更复杂的处理逻辑，比如回退以及一些其他异常的处理，而不是直接返回final_text
                # * 让planner重试，补一次 tool 输出（JSON已解析，但缺少tool字段）
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
                        plan, planner_trace = self.planner.plan_with_trace(planner_input)  # * 调用planner，返回plan和trace
                        self._record_planner(state, planner_input, copy.deepcopy(plan), planner_trace)
                        tool = plan.get("tool") if plan else None
                        args = plan.get("args", {}) if plan else {}
                        if tool:
                            break
                    # * 异常处理
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

            # * 应用预算控制，返回是否超出预算的最终结果
            budget_fail = self._apply_budget(state, tool, args)
            if budget_fail:
                # * 预算耗尽时尝试从已有状态输出
                final_obs = self._finalize_with_judge(state)
                if final_obs:
                    return {"final": final_obs, "trace": state.trace}
                return {"final": self._final_not_answerable(), "trace": state.trace}

            args = self._normalize_args(state, tool, args)
            if tool == "build_context" and not args.get("search_result"):
                return {"final": self._final_not_answerable(), "trace": state.trace}
            if tool == "answer" and not args.get("context"):
                return {"final": self._final_not_answerable(), "trace": state.trace}

            # * 调用对应的skill执行具体的工具逻辑
            # * 如果失败，给两次重试机会
            skill = get_skill(tool)
            max_retries = 2
            obs = None
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    obs = skill(args, self.ctx, self.llm_config)
                    break
                except Exception as exc:
                    last_exc = exc
                    state.trace.append({
                        "tool": "skill_error",
                        "args": {"tool": tool, "attempt": attempt + 1, "args": args},
                        "observation": {"error": str(exc)},
                    })
            if obs is None:
                return {
                    "final": {"final_text": f"Skill failed: {tool}: {last_exc}", "citations": []},
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
