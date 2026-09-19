"""Live tests: image-only PDFs route to vision (require GROQ_API_KEY).

Run: GROQ_API_KEY=... pytest -m live
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live
needs_key = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
ASSET = Path(__file__).parent / "assets" / "question_paper_455.pdf"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="testcase PDF not vendored")


@needs_key
@needs_asset
def test_live_image_only_pdf_falls_back_to_vision(tmp_path):
    """Rasterize one real paper page into an image-only PDF, then extract."""
    import fitz

    from paper_to_solution.pdf_ingest import extract_questions_from_pdf, pdf_to_page_texts

    src = fitz.open(ASSET)
    pix = src[1].get_pixmap(dpi=150)  # page 2: Q4-Q8, ApoExam table layout
    img_pdf = tmp_path / "image_only.pdf"
    with fitz.open() as out:
        for _ in range(1):
            page = out.new_page(width=pix.width * 72 / 150, height=pix.height * 72 / 150)
            page.insert_image(page.rect, pixmap=pix)
        out.save(img_pdf)

    texts = pdf_to_page_texts(img_pdf)
    assert all(len(t.strip()) < 300 for _, t in texts), "fixture must be image-only"

    res = extract_questions_from_pdf(img_pdf, mode="auto")
    assert len(res.questions) >= 3, f"expected Q4+ from vision, got {len(res.questions)}"
    numbers = [q.number for q in res.questions]
    assert any(n.strip().lstrip("Q") == "4" for n in numbers), f"numbers: {numbers}"
    assert all(q.page == 1 for q in res.questions)
    assert any(q.options for q in res.questions), "MCQ options should come through vision"
    # Canonical conformance: every vision question re-validates upstream-style.
    from paper_to_solution.canonical import Question as CanonicalQ
    for q in res.questions:
        CanonicalQ.model_validate(q.model_dump())
        assert q.type in ("mcq", "numerical", "short", "long")
