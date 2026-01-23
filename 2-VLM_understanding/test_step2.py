from summary import summarize_doc_json
DOC_NAME = "2310.08560v2"

PDF_PATH = "../demo/doc/" + DOC_NAME + ".pdf"

OUT_DIR = "../demo/imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../demo/jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")

INPUT_JSON = "../demo/jsons/" + DOC_NAME + "_ocr.json"
OUTPUT_JSON = "../demo/jsons/" + DOC_NAME + "_ocr_with_summary-new.json"

summarize_doc_json(
    input_json_path=INPUT_JSON,
    output_json_path=OUTPUT_JSON,
)

print(f"Done. Summarization output: {OUTPUT_JSON}")

from chunk_output import demo_build_blocks_for_doc_json

INPUT_JSON = "../demo/jsons/" + DOC_NAME + "_ocr_with_summary-new.json"
OUTPUT_JSON = "../demo/jsons/" + DOC_NAME + "_ocr_with_summary_chunks-new.json"
OUTPUT_DIR = "../demo/chunks-new/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

demo_build_blocks_for_doc_json(
    input_json_path=INPUT_JSON,
    output_json_path=OUTPUT_JSON,
    assets_root=OUTPUT_DIR,
)

print(f"Wrote: {OUTPUT_DIR}")