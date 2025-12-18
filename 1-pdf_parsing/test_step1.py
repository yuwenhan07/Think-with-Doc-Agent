from pdf_render import render_pdf_to_document
from ocr import OCRConfig, process_document_file
import os
import json

DOC_NAME = "2310.08560v2"

PDF_PATH = "../demo/doc/" + DOC_NAME + ".pdf"

OUT_DIR = "../demo/imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../demo/jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")
OUTPUT_JSON = "../demo/jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", "_ocr.json")

results = render_pdf_to_document(PDF_PATH, OUT_DIR, dpi=144, overwrite=True)

print(f"Rendered pages: {len(results)}")
print("First page:", results["pages"][0])

os.makedirs(os.path.dirname(JSON_DIR), exist_ok=True)
with open(JSON_DIR, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Wrote: {JSON_DIR}")

render_json_path = JSON_DIR
out_doc_path = OUTPUT_JSON

cfg = OCRConfig(prompt="qwenvl markdown")
doc = process_document_file(JSON_DIR, OUTPUT_JSON, cfg)
print(f"Done. OCR ok pages. Output: {out_doc_path}")