"""Offline tests: multi-page merge/dedup logic + LangGraph wiring (mocked LLM)."""
from paper_to_solution import graph as graph_mod
from paper_to_solution.canonical import Question
from paper_to_solution.extractor import parse_and_validate


def test_multipage_merge_dedupes_exact_repeats():
    # Simulate two pages where Q2 is repeated verbatim at the top of page 2
    seen: set[tuple[str, str]] = set()
    merged = []
    for page_no, raw in [
        (1, '{"questions": [{"question_number": "1", "question_text": "Q one"}, {"question_number": "2", "question_text": "Q two"}]}'),
        (2, '{"questions": [{"question_number": "2", "question_text": "Q two"}, {"question_number": "3", "question_text": "Q three"}]}'),
    ]:
        res = parse_and_validate(raw, model="t", default_page=page_no)
        for q in res.questions:
            q.source_page = page_no
            key = (q.question_number.strip(), q.question_text.strip()[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(q)
    assert [q.question_number for q in merged] == ["1", "2", "3"]
    assert [q.source_page for q in merged] == [1, 1, 2]


def test_empty_extraction_is_valid_but_empty():
    res = parse_and_validate('{"questions": []}', model="t")
    assert res.questions == []


def test_graph_solve_node_uses_answer_fn(monkeypatch):
    from paper_to_solution.schemas import SolvedQuestion

    def fake_answer(q):
        return SolvedQuestion(question_number=q.number, question_text=q.text,
                              answer=f"ANSWER:{q.number}", model="fake")

    monkeypatch.setattr(graph_mod, "answer_question", fake_answer)
    app = graph_mod.build_graph()
    out = app.invoke({"questions": [
        {"number": "1", "section": None, "text": "What is 2+2?", "marks": 2,
         "type": "numerical", "options": None, "has_figure": False,
         "page": 1, "choice_group": None},
    ], "solutions": [], "errors": []})
    assert out["solutions"][0]["answer"] == "ANSWER:1"
    assert out["errors"] == []


def test_graph_records_per_question_errors_without_aborting(monkeypatch):
    from paper_to_solution.schemas import SolvedQuestion

    def flaky(q):
        if q.number == "bad":
            raise RuntimeError("boom")
        return SolvedQuestion(question_number=q.number, question_text=q.text,
                              answer="ok", model="fake")

    monkeypatch.setattr(graph_mod, "answer_question", flaky)
    qs = [
        Question(number="1", text="fine"),
        Question(number="bad", text="broken"),
        Question(number="3", text="also fine"),
    ]
    out = graph_mod.run_graph(qs)
    assert len(out["solutions"]) == 2
    assert len(out["errors"]) == 1 and "bad" in out["errors"][0]


def test_graph_rejects_non_canonical_type():
    import pytest

    with pytest.raises(Exception):
        Question(number="1", text="x", type="descriptive")  # not in canonical literal
