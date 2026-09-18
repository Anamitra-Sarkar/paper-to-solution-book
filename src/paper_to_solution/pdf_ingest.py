"""PDF ingestion (Yugal): PDF -> page images -> reusable image extractor."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from paper_to_solution import config
from paper_to_solution.extractor import extract_questions_from_pil
from paper_to_solution.schemas import ExtractionResult


def pdf_to_page_images(pdf_path: str | Path, dpi: int = 200):
    """Render PDF pages to PIL images. Raises ValueError on invalid PDFs."""
    p = Path(pdf_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")
    try:
        doc = fitz.open(p)
    except Exception as e:
        raise ValueError(f"Cannot open PDF {pdf_path}: {e}") from e
    images = []
    with doc:
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi)
            import io as _io

            from PIL import Image as _Image

            img = _Image.open(_io.BytesIO(pix.tobytes("png")))
            images.append((i, img))
    return images


def extract_questions_from_pdf(pdf_path: str | Path, model: str | None = None) -> ExtractionResult:
    """PDF -> structured questions, one vision call per page, source_page preserved."""
    model = model or config.EXTRACTION_MODEL
    pages = pdf_to_page_images(pdf_path)
    merged = []
    raw_parts: list[str] = []
    for page_no, img in pages:
        result = extract_questions_from_pil(img, source_page=page_no, model=model)
        raw_parts.append(result.raw_response or "")
        merged.extend(result.questions)
    return ExtractionResult(questions=merged, raw_response="\n".join(raw_parts), model=model)
