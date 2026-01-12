# 4. Planner (LLM-driven)

这一阶段负责把「问题」编排成一个**可控的工具链执行流程**：
- 由 LLM Planner 输出下一步工具调用（tool + args）
- Executor 负责预算控制、状态记忆、强制评审、错误恢复
- 各个 skills 独立实现（rewrite/search/build_context/answer/judge/finalize）

目标是把检索与回答的流程变成**可复用、可追踪、可回放**的“计划-执行-裁判”闭环。

---

## **输入 / 依赖**
Planner 本身不直接读 PDF，它依赖上游阶段产物（来自 3-embedding）：

**索引与资产目录**
- `index_dir/summary.index.faiss`
- `index_dir/summary.meta.jsonl`
- `index_dir/blocks.manifest.json`（可选）
- `asset_base_dir/`（用于 bbox crop 的图片资产）

**运行时输入**
- 用户查询 `query`
- LLM API Key（默认环境变量 `QianFan_API_KEY`）

---

## **输出**
最终输出统一为：
```json
{
  "final_text": "...",
  "citations": [{"page": 2, "block_id": "p2:b0003", "quote": "..."}],
  "confidence": 0.72
}
```
同时返回完整 trace 便于回放：
```json
{
  "final": {...},
  "trace": [
    {"tool": "search", "args": {...}, "observation": {...}},
    {"tool": "judge_retrieval", "args": {...}, "observation": {...}}
  ]
}
```

---

## **核心流程（Planner + Executor）**
实际执行流程在 `executor.py` 中，关键逻辑如下：

1. **Planner 决策下一步工具**
   - `LLMPlanner.plan()` 输出 `{ "tool": "...", "args": {...} }`
   - 只能从固定工具集合里选

2. **Executor 进行预算与强制评审**
   - 每次 `search` 后强制 `judge_retrieval`
   - 每次 `answer` 后强制 `judge_answer`
   - 超预算时自动 fallback（如已有搜索结果则直接构建上下文回答）

3. **工具链基本顺序（常见路径）**
```
rewrite -> search -> judge_retrieval -> build_context -> answer -> judge_answer -> finalize
```

---

## **Planner 允许的 JSON Schema**
Planner 只能输出以下格式之一（LLM prompt 强制限制）：
```
{"tool": "rewrite", "args": {...}}
{"tool": "search", "args": {...}}
{"tool": "judge_retrieval", "args": {...}}
{"tool": "build_context", "args": {...}}
{"tool": "answer", "args": {...}}
{"tool": "judge_answer", "args": {...}}
{"tool": "finalize", "args": {...}}
{"final": {...}}
```

额外约束：
- `answer` 后必须 `judge_answer`
- `search` 后必须 `judge_retrieval`
- args 必须简短，不能塞全文内容

---

## **Executor 的预算控制（BudgetConfig）**
`executor.py` 默认预算：
- `max_turns`: 8
- `max_search_calls`: 3
- `max_rewrite_calls`: 2
- `max_blocks_context`: 8
- `max_same_query_search`: 2

超预算时处理：
- **search 超限**：若已有搜索结果，自动剪枝 + 直接回答
- **rewrite 超限**：直接停止 rewrite
- **turn 超限**：返回 `Turn budget exceeded.`

---

## **Skills 一览（职责 + 输入输出）**

### **rewrite**
目的：对 query 做意图识别 + 改写，输出更适合检索的 query 版本。

输入：
```json
{"query": "...", "mode": "theme", "max_rewrites": 3}
```
输出：
```json
{"intent": "paper_theme", "rewrites": ["..."], "negative": [], "page_prior": [], "notes": "..."}
```

### **search**
目的：两阶段检索（summary -> block），返回候选块 + 统计信息。支持 filters。

输入：
```json
{"query": "...", "k_pages": 8, "k_blocks": 30, "final_topk": 10, "filters": {"avoid_sections": ["references"]}}
```
输出：
```json
{"candidate_pages": [2, 6], "summary_hits": [...], "block_hits": [...], "stats": {"refs_ratio": 0.2}}
```

### **judge_retrieval**
目的：判断检索是否有效、是否命中核心内容，并给出下一步建议。

输出：
```json
{"verdict": "good|bad|uncertain", "reasons": [...], "suggestions": [{"action": "rewrite"}]}
```

### **build_context**
目的：从 block_hits 中挑选少量高价值证据，控制上下文大小。

输入：
```json
{"search_result": {...}, "intent": "paper_theme", "max_blocks": 8}
```
输出：
```json
{"context": {"question": "...", "evidence": [{"page": 2, "block_id": "p2:b0003", "text": "..."}]}}
```

### **answer**
目的：只基于 evidence 输出回答，并给出引用。

输入：
```json
{"context": {"question": "...", "evidence": [...]}, "need_citations": true, "style": "short"}
```
输出：
```json
{"answer": "...", "citations": [{"page": 2, "block_id": "p2:b0003", "quote": "..."}], "confidence": 0.72}
```

### **judge_answer**
目的：检查回答是否被证据支持，是否需要更多检索。

输出：
```json
{"verdict": "final|revise|need_more_evidence", "issues": [...], "next_actions": [...]}
```

### **finalize**
目的：把 answer 统一成最终输出结构。

输出：
```json
{"final_text": "...", "citations": [...], "confidence": 0.72}
```

---

## **运行方式**
```bash
export QianFan_API_KEY="bce-v3/..."
python run.py --query "Summarize the paper"
```

可选参数：
- `--index_dir`：默认 `../demo/index_out/2310.08560v2`
- `--asset_base_dir`：默认 `../demo/chunks/2310.08560v2`
- `--max_turns / --max_search_calls / --max_rewrite_calls / --max_blocks_context`

---

## **设计要点（与前几阶段衔接）**
- 4 阶段完全依赖 3 阶段输出索引（summary + blocks）。
- 通过 `build_context` 控制证据规模，避免回答时上下文膨胀。
- `judge_retrieval`/`judge_answer` 形成强制闭环，提高可靠性。
- 全 trace 可回放，适合做调试与自动评测。
