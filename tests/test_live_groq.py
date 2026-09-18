"""Live tests: real Groq Qwen vision + answer path. Require GROQ_API_KEY.

Run: GROQ_API_KEY=... pytest -m live
"""
import os

import pytest

from tests.helpers import make_paper_image

pytestmark = pytest.mark.live
needs_key = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


@needs_key
def test_live_single_clear_image(tmp_path):
    from paper_to_solution.extractor import extract_questions_from_image

    img = make_paper_image(tmp_path / "paper1.png", [
        "CLASS TEST - MATHEMATICS (10 marks)",
        "Q1. What is 7 x 8? [2 marks]",
        "Q2. Solve for x: 2x + 5 = 15. [3 marks]",
    ])
    res = extract_questions_from_image(img)
    assert len(res.questions) >= 1
    texts = " ".join(q.question_text for q in res.questions)
    assert "7" in texts or "x" in texts  # sanity: real content extracted


@needs_key
def test_live_mcq_numbering_marks(tmp_path):
    from paper_to_solution.extractor import extract_questions_from_image

    img = make_paper_image(tmp_path / "mcq.png", [
        "SCIENCE QUIZ",
        "Q1. Which planet is known as the Red Planet? [1 mark]",
        "(A) Venus  (B) Mars  (C) Jupiter  (D) Saturn",
        "Q2. (a) Define force. [2 marks]  (b) State its SI unit. [1 mark]",
    ])
    res = extract_questions_from_image(img)
    assert len(res.questions) >= 2
    q1 = res.questions[0]
    assert q1.options, "MCQ options should be extracted"


@needs_key
def test_live_multipage_source_pages(tmp_path):
    from paper_to_solution.extractor import extract_questions_from_images

    p1 = make_paper_image(tmp_path / "p1.png", ["PAGE ONE", "Q1. First question here? [2 marks]"])
    p2 = make_paper_image(tmp_path / "p2.png", ["PAGE TWO", "Q2. Second question here? [3 marks]"])
    res = extract_questions_from_images([p1, p2])
    pages = {q.source_page for q in res.questions}
    assert pages == {1, 2}


@needs_key
def test_live_end_to_end_graph_answer(tmp_path):
    from paper_to_solution.extractor import extract_questions_from_image
    from paper_to_solution.graph import run_graph

    img = make_paper_image(tmp_path / "e2e.png", ["Q1. What is 12 + 8? [2 marks]"])
    res = extract_questions_from_image(img)
    assert res.questions, "acceptance: at least one question extracted"
    final = run_graph(res.questions)
    assert final["solutions"], "acceptance: at least one answer generated"
    assert "20" in final["solutions"][0]["answer"]
