import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, Iterable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ===== 基本配置 =====
BASE_URL = "https://qianfan.baidubce.com/v2"
API_KEY = "bce-v3/ALTAK-zVZzcJJcznidBUa8U1qAg/090d7e36ae3320aa004c36d5ea0c7678d97de0a9"
DEFAULT_MODEL = "deepseek-v3.2"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ===== 工具函数 =====

def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 jsonl 文件"""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_bool_or_none(text: str):
    """
    将模型输出解析为 True / False / None。
    支持：correct / incorrect / true / false
    支持模型啰嗦输出，例如 "The answer is correct." 也能识别
    """
    if text is None:
        return None

    v = text.strip().lower()

    # 从文本中抽取关键 token
    m = re.search(r"\b(correct|incorrect|true|false)\b", v)
    if m:
        token = m.group(1)
        if token in ("true", "correct"):
            return True
        if token in ("false", "incorrect"):
            return False

    # 再做兜底匹配
    if "incorrect" in v or "false" in v:
        return False
    if "correct" in v or "true" in v:
        return True

    return text  # 无法解析，帮你保留原样以便 debug


def build_messages(rb: Dict[str, Any]):
    """
    适配新 jsonl 格式：
    - question      -> 问题
    - answer        -> 标准答案
    - pred_answer   -> 模型回答
    其它字段（doc_id, doc_type, evidence_xxx）目前不用。
    """
    question = rb.get("question")
    ground_truth = rb.get("answer")
    response = rb.get("final_text")

    sys_prompt = (
        "Your task is to evaluate whether the model's response correctly answers the question, "
        "based on the provided reference answer.\n"
        "This is part of an automated evaluation process, so your result must be STRICTLY either "
        "'correct' or 'incorrect'.\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Model Response: {response}\n\n"
        "Output only one word: correct or incorrect。"
    )

    sys_prompt = sys_prompt.replace("{question}", str(question))
    sys_prompt = sys_prompt.replace("{ground_truth}", str(ground_truth))
    sys_prompt = sys_prompt.replace("{response}", str(response))

    return [{"role": "system", "content": sys_prompt}]


# ===== 单条记录的评估函数（在线程里跑） =====

def eval_one(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    评估一条记录，在线程池中调用。
    idx 是这条记录在文件中的行号索引，用来保证输出顺序。
    """
    # 以前用的是 item.get("index")，新格式没有这个字段，
    # 这里优先用 doc_id，当成这条记录的 id；如果没有就退回 idx。
    rid = item.get("doc_id", idx)

    params = {
        "model": DEFAULT_MODEL,
        "messages": build_messages(item),
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 128,
    }

    try:
        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content or " "

        # 直接解析模型输出
        judge_val = to_bool_or_none(content)

        if judge_val is True:
            judge_str = "True"
        elif judge_val is False:
            judge_str = "False"
        else:
            judge_str = judge_val  # 无法解析则保留原样便于 debug

    except Exception as e:
        print(f"[ERROR] 调用模型失败 idx={idx}, id={rid}: {e}", file=sys.stderr)
        judge_str = None

    record = {
        "id": rid,
        "judge": judge_str,
    }
    return {
        "idx": idx,       # 保留原始顺序用
        "record": record,
    }


# ===== 主流程（并发版） =====

def run(input_path: str, output_path: str, max_workers: int = 5):
    in_p, out_p = Path(input_path), Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. 全部读入内存，带上行号（顺序）
    items: List[Dict[str, Any]] = list(read_jsonl(in_p))
    total = len(items)
    if total == 0:
        print("输入文件为空")
        return

    print(f"总共 {total} 条记录，使用 {max_workers} 个线程并发评估")

    # 结果列表，占位，后面用 idx 填充对应位置，保证输出顺序不乱
    results: List[Dict[str, Any]] = [None] * total

    # 2. 线程池并发调用
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(eval_one, item, idx): idx
            for idx, item in enumerate(items)
        }

        finished = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                r = future.result()
                results[idx] = r["record"]
            except Exception as e:
                print(f"[ERROR] 处理 idx={idx} 时异常: {e}", file=sys.stderr)
                # 即便异常，也占个坑，避免写文件时报 None
                rid = items[idx].get("doc_id", idx)
                results[idx] = {"id": rid, "judge": None}

            finished += 1
            if finished % 10 == 0 or finished == total:
                print(f"[{finished}/{total}] 条完成")

    # 3. 写出到文件（严格按原始顺序）
    with out_p.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(results):
            if rec is None:
                # 理论上不会发生，这里多一层保险
                rid = items[idx].get("doc_id", idx)
                rec = {"id": rid, "judge": None}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done: {total} lines → {out_p}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python judge_mt.py <输入answers.jsonl> <输出.jsonl> [max_workers]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if len(sys.argv) >= 4:
        try:
            max_workers = int(sys.argv[3])
        except ValueError:
            print("max_workers 必须是整数，已回退到默认 5")
            max_workers = 20
    else:
        max_workers = 20

    run(input_path, output_path, max_workers=max_workers)