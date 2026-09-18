"""Offline tests for the digital-text PDF path (no API calls).

Fixture: tests/assets/question_paper_455.pdf, the official testcase paper
(CBSE Class 9 Mathematics, 9 pages) from the answer-book test_papers set.
"""
from pathlib import Path

import pytest

from paper_to_solution.pdf_ingest import extract_questions_from_pdf, pdf_to_page_texts

ASSET = Path(__file__).parent / "assets" / "question_paper_455.pdf"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="testcase PDF not vendored")


@needs_asset
def test_extracts_all_37_questions_in_order():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert [q.question_number for q in res.questions] == [str(n) for n in range(1, 38)]


@needs_asset
def test_every_question_has_marks():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert all(q.marks is not None for q in res.questions)


@needs_asset
def test_mcq_options_extracted():
    res = extract_questions_from_pdf(ASSET, mode="text")
    q1 = res.questions[0]
    assert q1.question_type == "mcq"
    assert len(q1.options) == 4
    assert q1.options[0].startswith("(A)")
    assert q1.marks == 1


@needs_asset
def test_or_alternative_kept_with_question():
    res = extract_questions_from_pdf(ASSET, mode="text")
    q26 = next(q for q in res.questions if q.question_number == "26")
    assert "OR:" in q26.question_text
    assert "third term is 16" in q26.question_text
    assert q26.marks == 3


@needs_asset
def test_cross_page_continuation_merged():
    # Q31 starts on page 6; its OR alternative is on page 7.
    res = extract_questions_from_pdf(ASSET, mode="text")
    q31s = [q for q in res.questions if q.question_number == "31"]
    assert len(q31s) == 1
    assert q31s[0].source_page == 6
    assert "OR:" in q31s[0].question_text
    assert "street intersections" in q31s[0].question_text


@needs_asset
def test_subparts_split_main_and_or_separately():
    res = extract_questions_from_pdf(ASSET, mode="text")
    q35 = next(q for q in res.questions if q.question_number == "35")
    labels = [s.label for s in q35.subquestions]
    assert labels == ["i", "ii", "iii", "iv", "OR-i", "OR-ii", "OR-iii", "OR-iv"]


@needs_asset
def test_sections_and_chapters_recorded():
    res = extract_questions_from_pdf(ASSET, mode="text")
    by_no = {q.question_number: q for q in res.questions}
    assert by_no["34"].extraction_notes.startswith("Section D")
    assert "Ch 6" in by_no["34"].extraction_notes
    assert "digital text layer" in by_no["1"].extraction_notes


@needs_asset
def test_pages_cover_full_document():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert sorted({q.source_page for q in res.questions}) == list(range(1, 10))


@needs_asset
def test_page_texts_detect_digital_pages():
    texts = pdf_to_page_texts(ASSET)
    assert len(texts) == 9
    assert all(len(t.strip()) > 300 for _, t in texts)


@needs_asset
def test_auto_mode_uses_text_path_for_digital_pdf():
    res = extract_questions_from_pdf(ASSET, mode="auto")
    assert len(res.questions) == 37
    assert res.model == "text-parser"
    assert res.raw_response is None  # no vision calls were made
