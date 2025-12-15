import json
from pdf_render import render_pdf_to_document
import os

PDF_PATH = "../doc/2310.08560v2.pdf"

OUT_DIR = "../imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")

results = render_pdf_to_document(PDF_PATH, OUT_DIR, dpi=144, overwrite=True)

print(f"Rendered pages: {len(results)}")
print("First page:", results["pages"][0])

os.makedirs(os.path.dirname(JSON_DIR), exist_ok=True)
with open(JSON_DIR, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Wrote: {JSON_DIR}")