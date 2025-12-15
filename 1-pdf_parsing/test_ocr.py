from ocr import OCRConfig, ocr_document_pages, attach_ocr_results
import json

PDF_PATH = "../doc/2310.08560v2.pdf"

OUT_DIR = "../imgs/" + PDF_PATH.split("/")[-1].replace(".pdf", "")

JSON_DIR = "../jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", ".json")

OUTPUT_JSON = "../jsons/" + PDF_PATH.split("/")[-1].replace(".pdf", "_ocr.json")

render_json_path = JSON_DIR
out_doc_path = OUTPUT_JSON

with open(render_json_path, "r", encoding="utf-8") as f:
    render_manifest = json.load(f)

# Build a minimal doc structure compatible with attach_ocr_results
doc = {
    "pages": [
        {
            "page_number": int(p["page_number"]),
            "width_px": int(p.get("width_px", 0)),
            "height_px": int(p.get("height_px", 0)),
            "dpi": p.get("dpi"),
            "image_path": p.get("image_path"),
            "image_sha256": p.get("image_sha256"),
            "renderer": p.get("renderer"),
            "colorspace": p.get("colorspace"),
        }
        for p in render_manifest
    ]
}

cfg = OCRConfig()
ocr_results = ocr_document_pages(render_manifest, cfg)
doc = attach_ocr_results(doc, ocr_results)

with open(out_doc_path, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)

ok_pages = sum(1 for r in ocr_results if r.get("ok"))
print(f"Done. OCR ok pages: {ok_pages}/{len(ocr_results)}. Output: {out_doc_path}")