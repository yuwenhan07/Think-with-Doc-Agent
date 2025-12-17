from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageRenderResult:
    page_number: int
    dpi: int
    width_px: int
    height_px: int
    image_path: str
    image_sha256: str
    renderer: str = "pymupdf"
    colorspace: str = "rgb"

@dataclass(frozen=True)
class DocumentRenderResult:
    doc_id: str
    source: Dict[str, Any]
    num_pages: int
    metadata: Dict[str, Any]
    pages: List[PageRenderResult]


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_doc_id(pdf_path: str) -> str:
    """Stable document identifier derived from PDF bytes."""
    pdf_sha = _sha256_file(pdf_path)
    return f"sha256:{pdf_sha}"

def render_pdf_to_png(
    pdf_path: str,
    out_dir: str,
    dpi: int = 144,
    basename: str = "page",
    overwrite: bool = False,
) -> List[PageRenderResult]:
    """
    Render a PDF into per-page PNG images.
    Deterministic given (pdf_bytes, dpi, renderer version).
    """
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    pdf_path = os.path.abspath(pdf_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    results: List[PageRenderResult] = []

    # PyMuPDF uses a scaling matrix. 72 points per inch in PDF coordinate system.
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    for i in range(doc.page_count):
        page_number = i + 1
        page = doc.load_page(i)

        # Render to pixmap
        pix = page.get_pixmap(matrix=matrix, alpha=False)  # alpha=False => RGB background
        width_px, height_px = pix.width, pix.height

        filename = f"{basename}_{page_number:04d}.png"
        image_path = os.path.join(out_dir, filename)

        if (not overwrite) and os.path.exists(image_path):
            # If not overwriting, still compute hash for traceability
            image_sha = _sha256_file(image_path)
        else:
            pix.save(image_path)
            image_sha = _sha256_file(image_path)

        results.append(
            PageRenderResult(
                page_number=page_number,
                dpi=dpi,
                width_px=width_px,
                height_px=height_px,
                image_path=image_path,
                image_sha256=f"sha256:{image_sha}",
            )
        )

    doc.close()
    return results

def build_document_render_result(
    pdf_path: str,
    page_results: List[PageRenderResult],
    dpi: int,
    source: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DocumentRenderResult:
    """Lower-level helper: build a DocumentRenderResult. Prefer render_pdf_to_document() for one-step JSON output."""
    abs_path = os.path.abspath(pdf_path)

    if source is None:
        source = {
            "type": "local_file",
            "path": abs_path,
        }

    base_metadata: Dict[str, Any] = {
        "title": None,
        "author": None,
        "creation_date": None,
        "parser": {
            "renderer": "pymupdf",
            "dpi": dpi,
        },
    }
    if metadata:
        # Shallow-merge user provided metadata over defaults
        base_metadata.update(metadata)

    return DocumentRenderResult(
        doc_id=compute_doc_id(abs_path),
        source=source,
        num_pages=len(page_results),
        metadata=base_metadata,
        pages=page_results,
    )

def results_to_dict(results: List[PageRenderResult]) -> List[dict]:
    """Helper for JSON serialization."""
    return [asdict(r) for r in results]

def render_pdf_to_document(
    pdf_path: str,
    out_dir: str,
    dpi: int = 144,
    basename: str = "page",
    overwrite: bool = False,
    source: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """One-step API: render PDF pages and return a document-level JSON-serializable dict."""
    pages = render_pdf_to_png(
        pdf_path=pdf_path,
        out_dir=out_dir,
        dpi=dpi,
        basename=basename,
        overwrite=overwrite,
    )

    abs_path = os.path.abspath(pdf_path)

    if source is None:
        source = {
            "type": "local_file",
            "path": abs_path,
        }
    docname = os.path.basename(pdf_path)
    base_metadata: Dict[str, Any] = {
        "doc_name": docname,
        "parser": {
            "renderer": "pymupdf",
            "dpi": dpi,
        },
    }
    if metadata:
        base_metadata.update(metadata)

    return {
        "doc_id": compute_doc_id(abs_path),
        "source": source,
        "num_pages": len(pages),
        "metadata": base_metadata,
        "pages": [asdict(p) for p in pages],
    }