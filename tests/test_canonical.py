"""Offline tests: canonical conformance of the OCR/vision path (no API calls).

The canonical Question/Paper models are the team-wide contract shared with
PDF ingestion. These tests pin the converter behavior and the vendored
schema shape.
"""
import pytest

from paper_to_solution.canonical import (
    Paper,
    Question,
    canonical_type,
    to_canonical_paper,
    to_canonical_question,
)
from paper_to_solution.extractor import ExtractionError, parse_and_validate
from paper_to_solution.schemas import ExtractedQuestion


def _internal(**kw):
    base = {"question_number": "1", "question_text": "What is 2+2?"}
    base.update(kw)
    return ExtractedQuestion(**base)


def test_vendored_schema_field_names():
    assert set(Question.model_fields) == {
        "number", "section", "text", "marks", "type",
        "options", "has_figure", "page", "choice_group",
    }
    assert set(Paper.model_fields) == {
        "paper_id", "fingerprint", "status", "subject", "class_name", "board",
        "questions", "total_questions", "total_marks", "sections",
    }


def test_type_mapping_covers_all_internal_types():
    assert canonical_type("mcq") == "mcq"
    assert canonical_type("numerical") == "numerical"
    assert canonical_type("short_answer") == "short"
    assert canonical_type("descriptive") == "long"
    assert canonical_type("coding") == "long"
    assert canonical_type("fill_in_the_blank") == "short"
    assert canonical_type("true_false") == "short"
    assert canonical_type("unknown") == "short"
    assert canonical_type("bogus") == "short"


def test_options_empty_list_becomes_none():
    q = to_canonical_question(_internal(options=[]))
    assert q.options is None
    q2 = to_canonical_question(_internal(options=["(A) 3", "(B) 4"]))
    assert q2.options == ["(A) 3", "(B) 4"]


def test_marks_integral_float_becomes_int():
    assert to_canonical_question(_internal(marks=5.0)).marks == 5
    assert to_canonical_question(_internal(marks=5)).marks == 5
    assert to_canonical_question(_internal(marks=None)).marks is None


def test_choice_group_from_or_subquestions_and_text():
    q = to_canonical_question(_internal(
        question_text="Main part. OR Alternative part.",
        subquestions=[{"label": "OR-i", "text": "alt"}],
    ))
    assert q.choice_group == "1"
    q2 = to_canonical_question(_internal(question_text="Plain question."))
    assert q2.choice_group is None


def test_section_and_page_and_figure_pass_through():
    q = to_canonical_question(_internal(section="B", source_page=3, has_figure=True))
    assert (q.section, q.page, q.has_figure) == ("B", 3, True)


def test_full_text_preserved_and_validates():
    q = to_canonical_question(_internal(
        question_text="(i) Part one. (ii) Part two.", question_type="descriptive"))
    assert "(i)" in q.text and "(ii)" in q.text
    assert q.type == "long"
    Question.model_validate(q.model_dump())  # canonical re-validation


def test_paper_assembly_convention():
    qs = [to_canonical_question(_internal(question_number="1", marks=2, section="A")),
          to_canonical_question(_internal(question_number="2", marks=3, section="A"))]
    paper = to_canonical_paper(qs, [b"fake-bytes"])
    assert paper.paper_id == f"pap_{paper.fingerprint[:8]}"
    assert paper.status == "ready"
    assert paper.total_questions == 2
    assert paper.total_marks == 5
    assert paper.sections == ["A"]


def test_vision_new_fields_parsed():
    raw = '{"questions": [{"question_number": "3", "question_text": "See the figure.", "section": "C", "has_figure": true, "choice_group": null}]}'
    res = parse_and_validate(raw, model="t")
    q = res.questions[0]
    assert q.section == "C"
    assert q.has_figure is True
    out = to_canonical_question(q)
    assert (out.section, out.has_figure) == ("C", True)


def test_vision_choice_group_fallback_from_text():
    raw = '{"questions": [{"question_number": "5", "question_text": "Do this. OR Do that instead."}]}'
    res = parse_and_validate(raw, model="t")
    assert to_canonical_question(res.questions[0]).choice_group == "5"


def test_malformed_vision_output_still_rejected():
    with pytest.raises(ExtractionError):
        parse_and_validate("not json {{{", model="t")
    with pytest.raises(ExtractionError):
        parse_and_validate('{"questions": [{"question_text": "no number"}]}', model="t")


def test_vision_metadata_parsed_and_assembled():
    raw = ('{"metadata": {"subject": "Science", "class": "Class 9", "board": "CBSE"},'
           '"questions": [{"question_number": "1", "question_text": "Q?"}]}')
    res = parse_and_validate(raw, model="t")
    assert res.metadata == {"subject": "Science", "class": "Class 9", "board": "CBSE"}
    paper = to_canonical_paper([to_canonical_question(res.questions[0])], [b"x"],
                               res.metadata)
    assert (paper.subject, paper.class_name, paper.board) == ("Science", "Class 9", "CBSE")


def test_vision_metadata_absent_is_tolerated():
    res = parse_and_validate('{"questions": [{"question_number": "1", "question_text": "Q?"}]}',
                             model="t")
    assert res.metadata == {}
    paper = to_canonical_paper([to_canonical_question(res.questions[0])], [b"x"],
                               res.metadata)
    assert (paper.subject, paper.class_name, paper.board) == (None, None, None)


def test_text_header_metadata_captured():
    from paper_to_solution.text_parser import extract_paper_metadata

    pages = [(1, "Subject: Mathematics Part 1  |  Class: Class 9\n"
                 "Total Marks: 80  |  Time Allowed: 3h  |  CBSE\n")]
    assert extract_paper_metadata(pages) == {
        "subject": "Mathematics Part 1", "class": "Class 9", "board": "CBSE"}
    assert extract_paper_metadata([(1, "Q1. Something?")]) == {}


def test_tile_merge_dedupes_overlap_and_reassembles_straddlers():
    from paper_to_solution.extractor import _merge_tile_questions

    top = [_internal(question_number="4", question_text="Full question four."),
           _internal(question_number="5", question_text="First half of five")]
    bottom = [_internal(question_number="4", question_text="Full question four."),
              _internal(question_number="5", question_text="second half of five.")]
    merged = _merge_tile_questions(top, bottom)
    assert [q.question_number for q in merged] == ["4", "5"]
    q5 = merged[1]
    assert "First half" in q5.question_text and "second half" in q5.question_text
    assert "Reassembled" in (q5.extraction_notes or "")


def test_tile_merge_option_seen_by_either_tile_survives():
    """Regression: an MCQ option crossing the tile boundary must not be lost."""
    from paper_to_solution.extractor import _merge_tile_questions

    top = [_internal(question_number="7", question_text="Four-sided formula?",
                     options=["(A) 1", "(B) a", "(C) -1"])]
    bottom = [_internal(question_number="7", question_text="Four-sided formula?",
                        options=["(C) -1", "(D) 0"])]
    merged = _merge_tile_questions(top, bottom)
    assert len(merged) == 1
    assert merged[0].options == ["(A) 1", "(B) a", "(C) -1", "(D) 0"]
    # Same text seen twice: kept once, never duplicated.
    assert merged[0].question_text == "Four-sided formula?"
    assert "Reassembled" not in (merged[0].extraction_notes or "")


def test_tile_merge_substring_keeps_longer_without_duplication():
    from paper_to_solution.extractor import _merge_tile_questions

    top = [_internal(question_number="9", question_text="Find the value")]
    bottom = [_internal(question_number="9", question_text="Find the value of x when y is 2")]
    merged = _merge_tile_questions(top, bottom)
    assert len(merged) == 1
    assert merged[0].question_text == "Find the value of x when y is 2"


def test_find_split_row_avoids_text_lines():
    """Regression: tile split must fall in a whitespace gap, never through ink."""
    from PIL import Image, ImageDraw

    from paper_to_solution.extractor import _find_split_row

    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    ink_bands = [(60, 80), (140, 160), (240, 260), (320, 340)]  # fake text lines
    for y0, y1 in ink_bands:
        d.rectangle([20, y0, 380, y1], fill="black")
    split = _find_split_row(img, search_radius=120)
    assert 80 <= split <= 320  # inside the search band around mid=200
    assert all(not (y0 <= split <= y1) for y0, y1 in ink_bands), \
        f"split row {split} cuts through a text band"


def test_nested_tiling_recovers_after_tile_truncation(monkeypatch):
    """A tile that overflows is split again (depth 2 max); results merge."""
    from PIL import Image

    import paper_to_solution.extractor as ex

    calls = []

    def fake_internal(data_url, source_page, model):
        calls.append(source_page)
        from paper_to_solution.schemas import ExtractionResult
        if len(calls) <= 2:
            raise ex.TruncationError("overflow at full page and first tile")
        n = len(calls)
        q = _internal(question_number=str(n), question_text=f"Tile question {n}")
        return ExtractionResult(questions=[q], raw_response="{}", model=model,
                                metadata={})

    monkeypatch.setattr(ex, "_extract_internal", fake_internal)
    img = Image.new("RGB", (600, 1600), "white")
    res = ex._extract_page(img, 1, "test-model")
    assert len(res.questions) >= 2  # sub-tiles recovered via depth-2 recursion
    assert len(calls) == 5  # full + tile + 2 sub-tiles + second tile; capped depth
