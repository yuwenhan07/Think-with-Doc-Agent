# 3. RAG Embedding

**现有内容示例：**

```
{
  "doc_id": "sha256:9f674bcff69c86f11c813dcfad613d8841f5f8ed17979e3c4df06a91df7762e0",
  "source": {
    "type": "local_file",
    "path": "DocAgent/doc/2310.08560v2.pdf"
  },
  "num_pages": 13,
  "metadata": {
    "doc_name": "2310.08560v2.pdf",
    "parser": {
      "renderer": "pymupdf",
      "dpi": 144
    }
  },
  "pages": [
    {
      "page_number": 2,
      "dpi": 144,
      "width_px": 1224,
      "height_px": 1584,
      "image_path": "/Users/yuwenhan/Library/Mobile Documents/com~apple~CloudDocs/Documents/🐟/科研/Learn.agent/DocAgent/imgs/2310.08560v2/page_0002.png",
      "image_sha256": "sha256:81169f03539386c246a8f4f55f173309081b6009814f76236702520797d2503e",
      "renderer": "pymupdf",
      "colorspace": "rgb",
      "text_raw": "# MemGPT: Towards LLMs as Operating Systems\n\n<!-- Image (66, 75, 454, 256) -->\nFigure 1. MemGPT (left) writes data to persistent memory after it receives a system alert about limited context space.\n\n<!-- Image (514, 75, 898, 256) -->\nFigure 2. MemGPT (left) can search out-of-context data to bring relevant information into the current context window.\n\nwith *virtual memory*, which provides an illusion of there being more memory resources than are actually available in physical (i.e., main) memory by the OS paging overflow data to disk and retrieving data (via a page fault) back into memory when accessed by applications. To provide a similar illusion of longer context length (analogous to virtual memory), we allow the LLM to manage what is placed in its own context (analogous to physical memory) via an ‘LLM OS’, which we call MemGPT. MemGPT enables the LLM to retrieve relevant historical data missing from what is placed in-context, and also evict less relevant data from context and into external storage systems. Figure 3 illustrates the components of MemGPT.\n\nThe combined use of a memory-hierarchy, OS functions and event-based control flow allow MemGPT to handle unbounded context using LLMs that have finite context windows. To demonstrate the utility of our new OS-inspired LLM system, we evaluate MemGPT on two domains where the performance of existing LLMs is severely limited by finite context: document analysis, where the length of standard text files can quickly exceed the input capacity of modern LLMs, and conversational agents, where LLMs bound by limited conversation windows lack context awareness, persona consistency, and long-term memory during extended conversations. In both settings, MemGPT is able to overcome the limitations of finite context to outperform existing LLM-based approaches.\n\n## 2. MemGPT (MemoryGPT)\n\nMemGPT’s OS-inspired multi-level memory architecture delineates between two primary memory types: **main context** (analogous to main memory/physical memory/RAM) and **external context** (analogous to disk memory/disk storage). Main context consists of the LLM *prompt tokens*—anything in main context is considered *in-context* and can be accessed by the LLM processor during inference. External context refers to any information that is held outside of the LLMs fixed context window. This *out-of-context* data must always be explicitly moved into main context in order for it to be passed to the LLM processor during inference. MemGPT provides function calls that the LLM processor to manage its own memory without any user intervention.\n\n### 2.1. Main context (prompt tokens)\n\nThe prompt tokens in MemGPT are split into three contiguous sections: the **system instructions**, **working context**, and **FIFO Queue**. The system instructions are read-only (static) and contain information on the MemGPT control flow, the intended usage of the different memory levels, and instructions on how to use the MemGPT functions (e.g. how to retrieve out-of-context data). Working context is a fixed-size read/write block of unstructured text, writeable only via MemGPT function calls. In conversational settings, working context is intended to be used to store key facts, preferences, and other important information about the user and the persona the agent is adopting, allowing the agent to converse fluently with the user. The FIFO queue stores a rolling history of messages, including messages between the agent and user, as well as system messages (e.g. memory warnings) and function call inputs and outputs. The first index in the FIFO queue stores a system message containing a recursive summary of messages that have been evicted from the queue.\n\n### 2.2. Queue Manager\n\nThe queue manager manages messages in *recall storage* and the **FIFO queue**. When a new message is received by the system, the queue manager appends the incoming messages to the FIFO queue, concatenates the prompt tokens and triggers the LLM inference to generate LLM output (the completion tokens). The queue manager writes both the incoming message and the generated LLM output to recall storage (the MemGPT message database). When messages in recall storage are retrieved via a MemGPT function call, the queue manager appends them to the back of",
      "text_source": "ocr",
      "spans": [
        {
          "span_id": "2:ocr:region:0001",
          "type": "image",
          "bbox_rel": [
            0.05392156862745098,
            0.04734848484848485,
            0.3709150326797386,
            0.16161616161616163
          ],
          "bbox_px": [
            66.0,
            75.0,
            454.0,
            256.0
          ],
          "source": "ocr"
        },
        {
          "span_id": "2:ocr:region:0002",
          "type": "image",
          "bbox_rel": [
            0.4199346405228758,
            0.04734848484848485,
            0.7336601307189542,
            0.16161616161616163
          ],
          "bbox_px": [
            514.0,
            75.0,
            898.0,
            256.0
          ],
          "source": "ocr"
        }
      ],
      "diagnostics": {
        "ocr": {
          "model": "qwen3-vl-235b-a22b-instruct",
          "prompt": "qwenvl markdown",
          "min_pixels": 524288,
          "max_pixels": 4718592,
          "smart_resize": {
            "in_w": 1224,
            "in_h": 1584,
            "out_w": 1216,
            "out_h": 1600,
            "factor": 32
          },
          "elapsed_ms": 28275,
          "image_sha256": "sha256:81169f03539386c246a8f4f55f173309081b6009814f76236702520797d2503e"
        }
      },
      "page_summary": "- MemGPT is an OS-inspired system that lets LLMs manage memory like an operating system, using “main context” (in-context prompt tokens) and “external context” (out-of-context data stored externally) to simulate virtual memory and overcome finite context limits.\n- Figures 1 and 2 illustrate MemGPT’s core functions: writing data to persistent memory upon system alerts (Fig. 1) and retrieving relevant out-of-context data via search (Fig. 2) to maintain context during long conversations.\n- The main context is divided into three parts: read-only system instructions, a writable working context for key facts/preferences, and a FIFO queue that stores message history and system events, including recursive summaries of evicted messages.\n- The Queue Manager handles message flow by appending new messages to the FIFO queue, triggering LLM inference, and writing both input and output to recall storage (MemGPT’s message database).\n- MemGPT enables unbounded context handling for LLMs, improving performance in document analysis and conversational agents by dynamically managing memory without user intervention.\n- Section 2.2 begins describing the Queue Manager’s role but is cut off mid-sentence on this page.",
      "blocks": [
        {
          "block_id": "p2:b0001",
          "page_number": 2,
          "type": "figure",
          "span_id": "2:ocr:region:0001",
          "bbox_px": [
            80,
            120,
            552,
            410
          ],
          "asset_path": "../chunks/2310.08560v2/page_0002/p0002_2-ocr-region-0001.png",
          "crop_work_size": [
            1216,
            1600
          ],
          "text": "Figure 1. MemGPT (left) writes data to persistent memory after it receives a system alert about limited context space.",
          "source": "ocr_span"
        },
        {
          "block_id": "p2:b0002",
          "page_number": 2,
          "type": "figure",
          "span_id": "2:ocr:region:0002",
          "bbox_px": [
            625,
            120,
            1092,
            410
          ],
          "asset_path": "../chunks/2310.08560v2/page_0002/p0002_2-ocr-region-0002.png",
          "crop_work_size": [
            1216,
            1600
          ],
          "text": "Figure 2. MemGPT (left) can search out-of-context data to bring relevant information into the current context window.",
          "source": "ocr_span"
        },
        {
          "block_id": "p2:b0003",
          "page_number": 2,
          "type": "text",
          "text": "# MemGPT: Towards LLMs as Operating Systems",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0004",
          "page_number": 2,
          "type": "text",
          "text": "Figure 1. MemGPT (left) writes data to persistent memory after it receives a system alert about limited context space.",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0005",
          "page_number": 2,
          "type": "text",
          "text": "Figure 2. MemGPT (left) can search out-of-context data to bring relevant information into the current context window.",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0006",
          "page_number": 2,
          "type": "text",
          "text": "with *virtual memory*, which provides an illusion of there being more memory resources than are actually available in physical (i.e., main) memory by the OS paging overflow data to disk and retrieving data (via a page fault) back into memory when accessed by applications. To provide a similar illusion of longer context length (analogous to virtual memory), we allow the LLM to manage what is placed in its own context (analogous to physical memory) via an ‘LLM OS’, which we call MemGPT. MemGPT enables the LLM to retrieve relevant historical data missing from what is placed in-context, and also evict less relevant data from context and into external storage systems. Figure 3 illustrates the components of MemGPT.",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0007",
          "page_number": 2,
          "type": "text",
          "text": "The combined use of a memory-hierarchy, OS functions and event-based control flow allow MemGPT to handle unbounded context using LLMs that have finite context windows. To demonstrate the utility of our new OS-inspired LLM system, we evaluate MemGPT on two domains where the performance of existing LLMs is severely limited by finite context: document analysis, where the length of standard text files can quickly exceed the input capacity of modern LLMs, and conversational agents, where LLMs bound by limited conversation windows lack context awareness, persona consistency, and long-term memory during extended conversations. In both settings, MemGPT is able to overcome the limitations of finite context to outperform existing LLM-based approaches.",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0008",
          "page_number": 2,
          "type": "text",
          "text": "## 2. MemGPT (MemoryGPT)",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0009",
          "page_number": 2,
          "type": "text",
          "text": "MemGPT’s OS-inspired multi-level memory architecture delineates between two primary memory types: **main context** (analogous to main memory/physical memory/RAM) and **external context** (analogous to disk memory/disk storage). Main context consists of the LLM *prompt tokens*—anything in main context is considered *in-context* and can be accessed by the LLM processor during inference. External context refers to any information that is held outside of the LLMs fixed context window. This *out-of-context* data must always be explicitly moved into main context in order for it to be passed to the LLM processor during inference. MemGPT provides function calls that the LLM processor to manage its own memory without any user intervention.",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0010",
          "page_number": 2,
          "type": "text",
          "text": "### 2.1. Main context (prompt tokens)",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0011",
          "page_number": 2,
          "type": "text",
          "text": "The prompt tokens in MemGPT are split into three contiguous sections: the **system instructions**, **working context**, and **FIFO Queue**. The system instructions are read-only (static) and contain information on the MemGPT control flow, the intended usage of the different memory levels, and instructions on how to use the MemGPT functions (e.g. how to retrieve out-of-context data). Working context is a fixed-size read/write block of unstructured text, writeable only via MemGPT function calls. In conversational settings, working context is intended to be used to store key facts, preferences, and other important information about the user and the persona the agent is adopting, allowing the agent to converse fluently with the user. The FIFO queue stores a rolling history of messages, including messages between the agent and user, as well as system messages (e.g. memory warnings) and function call inputs and outputs. The first index in the FIFO queue stores a system message containing a recursive summary of messages that have been evicted from the queue.",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0012",
          "page_number": 2,
          "type": "text",
          "text": "### 2.2. Queue Manager",
          "source": "ocr_md_rule"
        },
        {
          "block_id": "p2:b0013",
          "page_number": 2,
          "type": "text",
          "text": "The queue manager manages messages in *recall storage* and the **FIFO queue**. When a new message is received by the system, the queue manager appends the incoming messages to the FIFO queue, concatenates the prompt tokens and triggers the LLM inference to generate LLM output (the completion tokens). The queue manager writes both the incoming message and the generated LLM output to recall storage (the MemGPT message database). When messages in recall storage are retrieved via a MemGPT function call, the queue manager appends them to the back of",
          "source": "ocr_md_rule"
        }
      ]
    }
]
}
```
* 至此，pdf初步处理已经完成了，一个pdf已经被拆分为  **图像、表格、文本**
* 每一页包含： **summary、chunks**

**下一步“**

基于已经切分好的chunks和summary做embedding，用于后续的检索。



## 规划
### **1) 定义“索引输入单元”与 ID 规范（先定接口，后面才稳）**
你现在 blocks 粒度已经够用，建议直接把每个 block 作为最小索引单元（后续再做 chunk merge 也不迟），并统一生成：

* chunk_id：建议沿用 block_id（如 p5:b0012），保证全局唯一
* modality：text | image | table
* payload：
    * text：text
    * image/table：asset_path（以及可选 caption/text，如果你愿意做“图像检索 + 文本过滤”）

* page_number、bbox_px、crop_work_size：用于回链与可视化定位
* source_doc_id：你已有 doc_id

这一步的产物是一个扁平化列表：chunks[]，而不是 page 嵌套结构（检索与向量库更友好）。

### **2) 计算 embedding（按你说的：文本不再处理；图/表用图像 embedding）**
建议策略：

* **Text blocks**：直接对 block.text 做 text embedding
* **Figure/Table blocks**：读取 asset_path 指向的裁剪图，做 image embedding
    * 若你最终希望“文本问图/表”，可以额外做一个轻量的 caption_embedding（用你已有的 block.text 或 page_summary），但这不是必须


落地结果：为每个 chunk_id 产出

* embedding: float[]
* embedding_model、dim、created_at
* modality

### **3) 建立向量索引与存储（先跑通，再优化）**
最低成本路线：

* 本地：FAISS（或 hnswlib）+ 一个 chunks_meta.jsonl
* 或你如果偏工程化：PostgreSQL + pgvector（和 MemGPT 的思路一致）

至少需要两个索引：

* text_index：只收 text embeddings
* vision_index：只收 image/table embeddings（这样查询时不会混淆空间；后续再做融合 re-ranking）

### **4) 做一个最小可用的检索 API（你很快就能验收）**
实现 3 个查询函数就能 Demo：

1. search_text(query, topk) → 返回 text blocks
2. search_image(query_image, topk) → 返回 figure/table blocks
3. search_hybrid(query_text, topk_text, topk_img) → 合并两路结果（简单加权或串联）

返回结果里必须带：

* chunk_id, page_number, bbox_px, asset_path/text这样你前端或 notebook 可以直接定位到图/表裁剪图，形成“可溯源”的 doc agent 体验。

### **5) 做“多模态融合”最小实现**
你说要“多模态 embedding 结合”，建议先用简单可靠的融合策略，不要一开始就上复杂模型：

* **Late Fusion（推荐）**：两路检索各取 topK，做归一化分数后加权：
    * score = w_text * sim_text + w_img * sim_img

* **Two-stage**：先文本召回（page/section），再在这些页内做图像召回（或反过来）

这一步不需要改你 Step2 的 JSON 结构，只在检索层做逻辑即可。



## 双索引方案：**目标索引设计**
### **A. Summary Index（粗召回 / 路由）**
**索引单元：page_summary（或你后续的 section_summary）**

* doc_id
* page_number
* summary_text（你 JSON 里已有 page_summary）
* embedding（用你的多模态 API，但这里输入就是纯文本）

用途：

* 作为第一阶段召回：先定位“可能相关的页/章节”
* 降低后续 chunk 检索的搜索空间，提高稳定性与速度

### **B. Chunk Index（细召回 / 证据）**
**索引单元：chunk（语义块），chunk 里包含文字 + 图像**

你现在的 JSON blocks 已经有 text / figure / table，建议先做一个“页面内聚合”的 chunk（不用做复杂语义分段，先跑通）：

**chunk 的组成（推荐 v1）：以 page 为边界聚合**

* chunk_id = f"p{page_number}:c0001"（每页一个 chunk，先最简单）
* text = page 的所有 text blocks 拼接（保持原顺序）
* images = 该页所有 figure/table 的 asset_path 列表
* page_number
* source_blocks = [block_id...]（用于回链）

**embedding 输入：**

* 如果你的多模态 API 支持“text + images”联合输入：直接一次调用得到一个向量。
* 如果只支持单输入：也可以做（text_embedding 与 image_embedding）再在你这边做融合，但你说 API 可多模态一致输出，我默认它支持联合输入。

用途：

* 用于第二阶段精检索：在候选页内找到最相关 chunk
* 返回 chunk 后可以再展开到 block 级（bbox、asset_path）做定位/高亮

## **检索流程**
1. 用 Summary Index 对 query 做 topK（比如 3~5 页）
2. 在这些页对应的 chunk 范围内，用 Chunk Index 再做 topK（比如 5~10）
3. 输出：chunk + 其关联 blocks（图/表裁剪图路径、bbox、页码）

这会非常稳，且易于后续扩展到“section summary / section chunks”。
