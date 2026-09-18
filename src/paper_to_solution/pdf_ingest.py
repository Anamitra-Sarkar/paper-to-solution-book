"""PDF ingestion: digital text layer + vision fallback per page (hybrid).

Strategy
--------
**auto** (default): each page is handled by the cheapest faithful path.

- Page carries a usable text layer  -> deterministic text parser
  (fast, exact, zero API cost).
- Page is scanned / image-only      -> Qwen vision extractor on the
  rendered page (same component as direct image uploads).

**vision**: force the vision extractor on every page, e.g. when diagrams and
figures must be interpreted even though text exists.
**text**: force the text parser on every page (pure offline path).

Every returned question records its provenance in ``extraction_notes``
("Source: digital text layer ..." vs "Source: vision OCR ...") together
with ``source_page``.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from paper_to_solution import config
from paper_to_solution.extractor import extract_questions_from_pil
from paper_to_solution.schemas import ExtractionResult
from paper_to_solution.text_parser import parse_text_pages

TEXT_FALLBACK_MIN_CHARS = 300


def pdf_to_page_images(pdf_path: str | Path, dpi: int = 200):
    """Render PDF pages to (page_number, PIL image) pairs."""
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

            images.append((i, _Image.open(_io.BytesIO(pix.tobytes("png")))))
    return images


def pdf_to_page_texts(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract (page_number, text) pairs from the PDF text layer."""
    p = Path(pdf_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")
    try:
        doc = fitz.open(p)
    except Exception as e:
        raise ValueError(f"Cannot open PDF {pdf_path}: {e}") from e
    out = []
    with doc:
        for i, page in enumerate(doc, start=1):
            out.append((i, page.get_text()))
    return out


def extract_questions_from_pdf(
    pdf_path: str | Path,
    model: str | None = None,
    mode: str = "auto",
) -> ExtractionResult:
    """PDF -> structured questions via the hybrid per-page strategy.

    Args:
        pdf_path: path to the question-paper PDF.
        model: vision model override (defaults to config.EXTRACTION_MODEL).
        mode: "auto" | "vision" | "text".
    """
    if mode not in ("auto", "vision", "text"):
        raise ValueError(f"Unknown mode {mode!r}: expected 'auto', 'vision' or 'text'")
    model = model or config.EXTRACTION_MODEL

    if mode == "vision":
        pages = pdf_to_page_images(pdf_path)
        merged, raw_parts = [], []
        for page_no, img in pages:
            result = extract_questions_from_pil(img, source_page=page_no, model=model)
            raw_parts.append(result.raw_response or "")
            merged.extend(result.questions)
        return ExtractionResult(questions=merged, raw_response="\n".join(raw_parts), model=model)

    texts = pdf_to_page_texts(pdf_path)
    if mode == "text":
        merged = parse_text_pages(texts)
        return ExtractionResult(questions=merged, raw_response=None, model="text-parser")

    # auto: text layer where usable, vision where the page is image-only.
    thin_pages = {n for n, t in texts if len(t.strip()) < TEXT_FALLBACK_MIN_CHARS}
    merged = parse_text_pages([(n, t) for n, t in texts if n not in thin_pages])
    raw_parts: list[str] = []
    if thin_pages:
        images = dict(pdf_to_page_images(pdf_path))
        vision_qs = []
        for n in sorted(thin_pages):
            result = extract_questions_from_pil(images[n], source_page=n, model=model)
            raw_parts.append(result.raw_response or "")
            vision_qs.extend(result.questions)
        # Merge keeping page order.
        by_page: dict[int, list] = {}
        for q in list(merged) + vision_qs:
            by_page.setdefault(q.source_page, []).append(q)
        merged = [q for n in sorted(by_page) for q in by_page[n]]
    return ExtractionResult(
        questions=merged,
        raw_response="\n".join(raw_parts) if raw_parts else None,
        model=f"text-parser+{model}" if raw_parts else "text-parser",
    )
