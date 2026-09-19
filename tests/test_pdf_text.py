"""Offline tests for the digital-text PDF path (no API calls).

Fixture: tests/assets/question_paper_455.pdf, the official testcase paper
(CBSE Class 9 Mathematics, 9 pages) from the answer-book test_papers set.

All assertions use the canonical Question/Paper schema shared with PDF
ingestion (number/section/text/marks/type/options/has_figure/page/choice_group).
"""
from pathlib import Path

import pytest

from paper_to_solution.canonical import Paper, Question
from paper_to_solution.pdf_ingest import extract_questions_from_pdf, pdf_to_page_texts

ASSET = Path(__file__).parent / "assets" / "question_paper_455.pdf"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="testcase PDF not vendored")


@needs_asset
def test_result_is_canonical_paper():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert isinstance(res, Paper)
    assert all(isinstance(q, Question) for q in res.questions)
    assert res.status == "ready"
    assert res.total_questions == 37
    assert res.paper_id == f"pap_{res.fingerprint[:8]}"
    assert res.sections == ["A", "B", "C", "D", "E"]


@needs_asset
def test_extracts_all_37_questions_in_order():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert [q.number for q in res.questions] == [str(n) for n in range(1, 38)]


@needs_asset
def test_every_question_has_marks():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert all(q.marks is not None for q in res.questions)


@needs_asset
def test_mcq_options_extracted():
    res = extract_questions_from_pdf(ASSET, mode="text")
    q1 = res.questions[0]
    assert q1.type == "mcq"
    assert len(q1.options) == 4
    assert q1.options[0].startswith("(A)")
    assert q1.marks == 1


@needs_asset
def test_or_alternative_kept_with_question_and_flagged():
    res = extract_questions_from_pdf(ASSET, mode="text")
    q26 = next(q for q in res.questions if q.number == "26")
    assert " OR " in q26.text
    assert "third term is 16" in q26.text
    assert q26.marks == 3
    assert q26.choice_group == "26"


@needs_asset
def test_cross_page_continuation_merged():
    # Q31 starts on page 6; its OR alternative is on page 7.
    res = extract_questions_from_pdf(ASSET, mode="text")
    q31s = [q for q in res.questions if q.number == "31"]
    assert len(q31s) == 1
    assert q31s[0].page == 6
    assert " OR " in q31s[0].text
    assert "street intersections" in q31s[0].text
    assert q31s[0].choice_group == "31"


@needs_asset
def test_subparts_folded_into_canonical_text():
    # Canonical schema has no subquestions list; parts stay in full text.
    res = extract_questions_from_pdf(ASSET, mode="text")
    q35 = next(q for q in res.questions if q.number == "35")
    assert "(i)" in q35.text and "(iv)" in q35.text and " OR " in q35.text


@needs_asset
def test_sections_single_letter_like_upstream():
    res = extract_questions_from_pdf(ASSET, mode="text")
    by_no = {q.number: q for q in res.questions}
    assert by_no["34"].section == "D"
    assert by_no["1"].section == "A"
    assert by_no["21"].section == "B"


@needs_asset
def test_pages_cover_full_document():
    res = extract_questions_from_pdf(ASSET, mode="text")
    assert sorted({q.page for q in res.questions}) == list(range(1, 10))


@needs_asset
def test_page_texts_detect_digital_pages():
    texts = pdf_to_page_texts(ASSET)
    assert len(texts) == 9
    assert all(len(t.strip()) > 300 for _, t in texts)


@needs_asset
def test_auto_mode_matches_text_mode_for_digital_pdf():
    res = extract_questions_from_pdf(ASSET, mode="auto")
    assert len(res.questions) == 37
    assert [q.number for q in res.questions] == [str(n) for n in range(1, 38)]
