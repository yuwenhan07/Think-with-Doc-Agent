from ocr import OCRConfig, process_document_file

PDF_PATH = "../demo/doc/2310.08560v2.pdf"

OUT_DIR = "../demo/imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../demo/jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")

OUTPUT_JSON = "../demo/jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", "_ocr.json")

render_json_path = JSON_DIR
out_doc_path = OUTPUT_JSON

cfg = OCRConfig(prompt="qwenvl markdown")
doc = process_document_file(JSON_DIR, OUTPUT_JSON, cfg)
print(f"Done. OCR ok pages. Output: {out_doc_path}")