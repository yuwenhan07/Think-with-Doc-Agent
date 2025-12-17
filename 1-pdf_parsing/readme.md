# 1. 文档接入与解析

**输入：**

pdf文档

**输出：**

在“文档接入与解析”这个阶段，你的输出目标不是“理解”，而是**把 PDF 稳定、可追溯、可复用地转成统一的 Page Object 集合 + 文档级元信息**。我建议把输出分成三层：**原始层（可复现）→ 解析层（可检索）→ 结构层（给后续布局/VLM 用）**。

## **最小必要输出（MVP）**
输入一个 PDF，你至少要输出：

### **A. 文档级对象 Document**
* doc_id：稳定标识（hash 或 UUID）
* source：文件名、路径、导入时间
* num_pages
* metadata：能取到多少取多少（取不到为空）
    * title / author / subject / creator / producer / creation_date / mod_date

* pages[]：按页顺序的 Page 列表（关键资产）

### **B. 页面级对象Page**
每页至少：

* page_number（从 1 开始）
* image：该页渲染后的图片路径（给 VLM / 视觉解析用）
* text_raw：该页原始文本（优先 text layer；没有则 OCR）
* text_source："pdf_text" 或 "ocr" 或 "mixed"
* page_size：宽高（points 或 pixels，需标单位）
* rotation：页面旋转信息（后面做 bbox 必须）

这套输出已经足够支持后续：

* 语义分块（基于 text_raw）
* 引用回跳（基于 page_number）
* 多模态理解（基于 image）

## **推荐的“工程化完整输出”**
在 MVP 之上，我强烈建议多输出两类信息：**可追溯的抽取证据**和**粗粒度布局线索**。

### **C. 抽取证据与质量信息（用于调试与鲁棒性）**
* extract_diagnostics（文档级）
    * 每页：has_text_layer、ocr_confidence_avg、render_dpi、text_char_count、image_sha256

* errors / warnings（例如某页渲染失败、OCR 超时等）

### **D. 粗分块（Pre-chunk）——不是语义 chunk，只是“方便检索/对齐”**
每页给一个粗粒度切分：

* spans：按段落/行切（来自 PDF text 或 OCR 结果）
    * span_id
    * text
    * bbox（可选但强烈建议：即便粗，也要有）
    * source（pdf_text/ocr）


## **输出格式（示例 JSON）**
```
{
  "doc_id": "string", 
  "source": {
    "type": "local_file | url | upload | database",
    "path": "string",
    "uri": "string?"
  },
  "num_pages": "int",
  "metadata": {
    "doc_name": "string | null",
    "parser": {
      "renderer": "string",
      "dpi": "int",
      "version": "string?"
    }
  },
  "pages": [
    "<PageObject>",
    "<PageObject>"
  ]
}
```
## **关键设计选择（现在就定下来，避免返工）**
* **页面图像是强制输出**：即使 PDF 有文本层，也要渲染图像，后续 VLM/布局/图表都依赖它。
* **文本来源要显式标记**：否则你无法解释“为什么这页搜不到/抽取错了”。
* **保留坐标系信息**：page_size、rotation、dpi；后续 bbox 映射离不开。
* **输出要可复现**：渲染参数（dpi、renderer）、OCR 参数（语言、引擎）写进 diagnostics。

