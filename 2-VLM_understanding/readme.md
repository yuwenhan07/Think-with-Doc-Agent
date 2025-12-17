2. VLM Understanding and Make Chunk

**现有结果：**

pdf parsing后的内容，一个json文件，包含每个pdf的基本信息，每一页ocr后的markdown结果，每一页的图片、表格的bbox（如有）

**进一步处理：**

希望把结果升级成 **block-level **的中间表示：

* 把“整页文本”拆成可操作的内容单元（标题/段落/列表/表格/图片/公式/页眉页脚等）。
* 给每个内容单元一个稳定的 block_id，并保留溯源信息（page_number、在原文中的位置、关联的 span）。
* 把图表这类非文本对象也纳入同一套 blocks，并尽量绑定它的 caption/邻近描述。

**最终每一页包含：**

1. 一个page level的summary
2. 基于ocr的结果，对现有文本进行分块，用于后续的embedding检索
3. block包含 文本、image/table 文本就以text形式给出，image table用bbox图片形式给出



## **目标输出（每页新增字段）**
### **Page-level**
* page_summary: 一段短摘要（建议 2–5 句，100–200 中文字/或 60–120 英文词）
* blocks: 用于 embedding 检索的最小单元列表（text / figure / table）

### **Block-level（统一 schema）**
每个 block 至少包含：

* block_id: 稳定 ID（例如 "p2:b003"）
* page_number
* type: "text" | "figure" | "table"
* text: 文本块的内容；表格块可放 caption/latex/ocr 表格文本（如果有）
* bbox_px / bbox_rel: 对 figure/table 必填（来自 spans）
* asset_path: 对 figure/table 必填（bbox crop 后的小图路径）
* meta: 可选（heading 层级、caption、char_range、span_id 等）

## **Step 2 的实现思路（按执行顺序）**
### **1) 生成 page-level summary（基于 OCR 文本）**
输入：page.text_raw

输出：page.page_summary

推荐做法：

* **第一版（最快）**：直接用 LLM 对 text_raw 总结（如果该页主要是图表/表格，summary 也要提到“本页包含 Figure/Table X，主题是什么”）。
* **注意点**：text_raw 里可能有大量 References 或长表格 latex；建议 summary 前先做一个轻量清洗：
    * 去掉 \begin{tabular}...\end{tabular} 大段（或截断到前 N 行）
    * 去掉连续的参考文献条目（References 页可只总结“本页为参考文献列表”）


验收：每页都有 page_summary，且能体现该页主题/结构。

### **2) 文本分块（用于 embedding 检索）**
输入：page.text_raw

输出：若干 type="text" 的 blocks

建议用**markdown 规则分块**（你 OCR 已经输出 markdown，适配度很高）：

* 以空行分段：形成段落单元
* 段落若以 # / ## / ### 开头：作为“heading block”（你也可以仍用 type="text"，在 meta.level 记录层级）
* 列表项（- / * / 1.）可合并为一个 text block，避免过碎
* 对于包含 \begin{tabular} 的段落：不要当作普通 text；交给 table block（见下一步）

**分块粒度建议：**

* 目标块大小：英文 80–250 tokens/块（或中文 150–500 字/块）
* 过长段落：按句号/分号再切一刀
* 过短段落：与上下相邻段落合并（尤其是被 OCR 误断行的情况）

验收：文本块数量合理（每页通常 5–20 个，视密度），且每块语义相对自包含。

### **3) 将 image/table 转成“可检索资产块”（bbox crop + 绑定文字）**
输入：page.spans[] + page.image_path + page.text_raw

输出：type="figure" / type="table" 的 blocks

你已经有 spans（含 bbox），还缺两件事：

1. **把 bbox 区域 crop 出来并落盘**（作为 asset_path）
2. **把与之相关的文字挂到该 block 上**（caption/邻近描述/表格 latex）

#### **3.1 bbox crop（强烈建议做）**
* 对每个 span：
    * 从 page.image_path 读取整页 png
    * 用 bbox_px=[x1,y1,x2,y2] crop
    * 存到固定目录：assets/<doc_id>/p{page_number}_{span_id}.png
    * 写回 block 的 asset_path


这样后续 embedding / rerank / VLM 复读都很顺。

#### **3.2 文字绑定（先做简单规则，效果就很好）**
* **Figure caption 绑定**：你 OCR 的格式非常典型：
    * <!-- Image (...) -->
    * Figure N. ...

* 规则：遇到 <!-- Image ... --> 注释行，就把“其后的第一段以 Figure 开头的文本”当作 caption，绑定给最近的 image span（按出现顺序一一对应即可）。
* **Table 绑定**：
    * 你这篇里表格往往是：## Table 1... + <!-- Table (...) --> + \begin{tabular}...
    * 规则：遇到 <!-- Table ... -->，向上找最近的 Table N. 标题/段落作为 caption；向下收集紧随其后的 \begin{tabular}...\end{tabular} 作为 table 的 text（哪怕是 latex，也比空白强，后续再结构化）。


验收：figure/table block 里既有 asset_path（可看图），也有 text（caption/latex/描述），检索与回答都能引用它。

## **你要的“block 里怎么放 text / image / table”**
按你的要求，“image/table 用 bbox 图片形式给出”，建议这样落地：

* type="text"：text 为正文文本；无 bbox/asset_path
* type="figure"：
    * asset_path：crop 后图片路径
    * bbox_px：原页 bbox
    * text：caption（如 “Figure 6. …”）

* type="table"：
    * asset_path：crop 后表格图片路径（用于 VLM/人眼验证）
    * bbox_px
    * text：优先放表格 caption + 表格 latex（或 OCR 表格文本）


## **最小 JSON 示例（单页）**
```
{
  "page_number": 2,
  "page_summary": "本页展示 MemGPT 的两个示意图，并引出虚拟内存类比与 MemGPT 组件概览；随后进入第 2 章并解释 main context / external context 的定义与队列结构。",
  "blocks": [
    {
      "block_id": "p2:b001",
      "type": "figure",
      "span_id": "2:ocr:region:0001",
      "bbox_px": [66,75,454,256],
      "asset_path": ".../assets/.../p2_2-ocr-region-0001.png",
      "text": "Figure 1. MemGPT (left) writes data to persistent memory..."
    },
    {
      "block_id": "p2:b002",
      "type": "figure",
      "span_id": "2:ocr:region:0002",
      "bbox_px": [514,75,898,256],
      "asset_path": ".../assets/.../p2_2-ocr-region-0002.png",
      "text": "Figure 2. MemGPT (left) can search out-of-context data..."
    },
    {
      "block_id": "p2:b003",
      "type": "text",
      "text": "MemGPT enables the LLM to retrieve relevant historical data..."
    }
  ]
}
```
