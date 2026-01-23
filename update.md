我看完了你的流水线（render → OCR → page summary → blocks → embedding → planner）。下面是按优先级的优化建议，都是基于你当前代码的具体位置。

Priority Fixes

修正 bbox 坐标链路不一致：ocr.py 里用页面尺寸计算 bbox_rel，但 chunk_output.py 在 smart_resize 时把 bbox_px 当 0–1000 归一化，导致 crop 错位；建议显式存 bbox_coord_type/bbox_source_size 并统一用 bbox_rel * work_size 裁剪。
规划器提示与工具白名单冲突：llm_planner.py 里允许的输出没有 judge_retrieval/judge_answer，但约束又要求它们；要么把这两个工具加入 allowed outputs，要么移除约束并完全靠 Executor 强制。
rewrite 产物没被真正利用：skill.py prompt 会生成“答案草稿”，而 executor.py 没把 rewrites 用于检索；建议把 rewrites 作为多查询检索入口并做结果合并。
证据污染风险：skill.py 会把整页 page_image_path 发给模型，可能读到未入证据的内容；建议仅发送 asset_path，或先为 text block 生成 bbox crop。
Retrieval & Provenance

增加文本层优先策略：目前纯 OCR，长文档错误率会高；建议在 pdf_render.py 或新增模块提取 PDF text layer，写 text_source="pdf_text|ocr|mixed" 并保留置信度。
建立 section/doc 级索引：summary.py 已给出 page_section，可以聚合成 section summary，再在 search.py 里作为候选页召回层，长文档更稳。
引入混合检索与重排：在 search.py 加 BM25/keyword 通道并与向量结果合并，再用交叉编码或 LLM rerank，提升精确词/公式召回。
上下文选择去重与多样性：skill.py 现在只按 score 选 top-k，容易聚集在同一页且重复 caption；建议按页分桶、去重、并补邻近 block。
Scale & Ops

Embedding 缓存与批量化：embedding.py 逐块调用外部 API，长文档成本高；建议按文本/图片 hash 缓存、批量请求、指数退避。
数据结构稳定性：为 text block 增加 char_range/line_no 或行级 bbox，保证 citations 可回溯；并在 chunk_output.py 固化 block_id 规则。
产物格式改成 JSONL 分片：长文档不应每次读写完整 JSON；建议 blocks/meta 按页或 section 写 JSONL 并增量索引。
加最小评测闭环：抽几个 QA 样本做 recall@k 和 citation 命中率，防止改动退化。
如果你希望我直接落地改动，我建议按下面顺序做（你选一个）：

修复 bbox 坐标与 crop 逻辑
规划器提示 + rewrite 多查询检索联动
混合检索 + build_context 去重与多样性