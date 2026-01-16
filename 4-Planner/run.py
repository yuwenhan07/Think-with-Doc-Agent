"""
run.py
命令行入口 + 配置文件 + 跑一次执行
query: 用户输入的查询
index_dir: 索引文件夹路径
asset_base_dir: 资源文件夹路径
max_turns: 最大对话轮数
max_search_calls: 最大搜索调用次数
max_rewrite_calls: 最大重写调用次数
max_blocks_context: 最大上下文块数
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from executor import BudgetConfig, Executor
# planners 和 skills 是两个文件夹，分别在两个文件夹的 __init__.py 中注册了对应的包与函数，可以直接调用
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

    # * Executor 是执行器，负责执行整个对话流程，传入配置参数，包括索引路径、资源路径、规划器配置、LLM配置和预算配置
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
