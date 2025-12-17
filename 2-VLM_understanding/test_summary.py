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