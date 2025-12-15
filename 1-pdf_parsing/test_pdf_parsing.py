from pdf_render import render_pdf_to_document
from ocr import OCRConfig, process_document_file


DOC_NAME = "2310.08560v2"

PDF_PATH = "../doc/" + DOC_NAME + ".pdf"

OUT_DIR = "../imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")

OUTPUT_JSON = "../jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", "_ocr.json")

results = render_pdf_to_document(PDF_PATH, OUT_DIR, dpi=144, overwrite=True)

print(f"Rendered pages: {len(results)}")
print("First page:", results["pages"][0])

render_json_path = JSON_DIR
out_doc_path = OUTPUT_JSON

cfg = OCRConfig(prompt="qwenvl markdown")
doc = process_document_file(JSON_DIR, OUTPUT_JSON, cfg)
print(f"Done. OCR ok pages. Output: {out_doc_path}")