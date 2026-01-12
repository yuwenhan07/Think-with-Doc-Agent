from __future__ import annotations

import argparse
import json
from pathlib import Path

from executor import BudgetConfig, Executor
from planners import PlannerConfig
from skills import LLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--index_dir", type=str, default="../demo/index_out/2310.08560v2")
    parser.add_argument("--asset_base_dir", type=str, default="../demo/chunks/2310.08560v2")
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--max_search_calls", type=int, default=3)
    parser.add_argument("--max_rewrite_calls", type=int, default=2)
    parser.add_argument("--max_blocks_context", type=int, default=8)
    args = parser.parse_args()

    planner_cfg = PlannerConfig()
    llm_cfg = LLMConfig()
    budget = BudgetConfig(
        max_turns=args.max_turns,
        max_search_calls=args.max_search_calls,
        max_rewrite_calls=args.max_rewrite_calls,
        max_blocks_context=args.max_blocks_context,
    )

    executor = Executor(
        index_dir=Path(args.index_dir),
        asset_base_dir=Path(args.asset_base_dir),
        planner_config=planner_cfg,
        llm_config=llm_cfg,
        budget=budget,
    )

    result = executor.run(args.query)
    print("\n=== Final Result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    trace = result.get("trace", [])
    if trace:
        print("\n=== Planner/Executor Trace ===")
        for i, step in enumerate(trace, 1):
            tool = step.get("tool")
            args_dump = json.dumps(step.get("args", {}), ensure_ascii=False)
            obs_dump = json.dumps(step.get("observation", {}), ensure_ascii=False)
            print(f"[{i}] tool={tool} args={args_dump}")
            print(f"    observation={obs_dump}")


if __name__ == "__main__":
    main()
