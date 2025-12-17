from summary import summarize_doc_json
DOC_NAME = "2310.08560v2"

PDF_PATH = "../doc/" + DOC_NAME + ".pdf"

OUT_DIR = "../imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")

INPUT_JSON = "../jsons/" + DOC_NAME + "_ocr.json"
OUTPUT_JSON = "../jsons/" + DOC_NAME + "_ocr_with_summary.json"

summarize_doc_json(
    input_json_path=INPUT_JSON,
    output_json_path=OUTPUT_JSON,
)

print(f"Done. Summarization output: {OUTPUT_JSON}")

from chunk_output import demo_build_blocks_for_doc_json

INPUT_JSON = "../jsons/" + DOC_NAME + "_ocr_with_summary.json"
OUTPUT_JSON = "../jsons/" + DOC_NAME + "_ocr_with_summary_chunks.json"
OUTPUT_DIR = "../chunks/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

demo_build_blocks_for_doc_json(
    input_json_path=INPUT_JSON,
    output_json_path=OUTPUT_JSON,
    assets_root=OUTPUT_DIR,
)

print(f"Wrote: {OUTPUT_DIR}")