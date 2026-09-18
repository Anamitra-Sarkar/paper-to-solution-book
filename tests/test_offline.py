"""Offline unit tests: schemas, validation, image handling (no API calls)."""
import io

import pytest
from PIL import Image

from paper_to_solution.extractor import ExtractionError, parse_and_validate
from paper_to_solution.image_io import ImageError, detect_mime, load_image_bytes, preprocess_image
from paper_to_solution.schemas import ExtractionResult


def test_parse_valid_mcq_and_subquestions():
    raw = """{"questions": [
      {"question_number": "1", "question_text": "What is 2+2?", "marks": 2,
       "question_type": "mcq", "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
       "subquestions": [], "source_page": 1, "confidence": 0.99, "extraction_notes": null},
      {"question_number": "2(a)", "question_text": "Define photosynthesis.", "marks": 3,
       "question_type": "short_answer", "options": [],
       "subquestions": [{"label": "a", "text": "Where does it occur?", "marks": 1}],
       "source_page": 1, "confidence": 0.9, "extraction_notes": null}
    ]}"""
    res = parse_and_validate(raw, model="test")
    assert isinstance(res, ExtractionResult)
    assert len(res.questions) == 2
    assert res.questions[0].options == ["A) 3", "B) 4", "C) 5", "D) 6"]
    assert res.questions[0].question_type == "mcq"
    assert res.questions[1].subquestions[0].label == "a"
    assert res.questions[1].marks == 3


def test_missing_optional_fields_normalized_to_null():
    raw = '{"questions": [{"question_number": "3", "question_text": "Explain gravity."}]}'
    res = parse_and_validate(raw, model="test")
    q = res.questions[0]
    assert q.marks is None
    assert q.options == []
    assert q.subquestions == []
    assert q.extraction_notes is None
    assert q.question_type == "unknown"


def test_unknown_question_type_falls_back_to_unknown():
    raw = '{"questions": [{"question_number": "1", "question_text": "x", "question_type": "essay"}]}'
    res = parse_and_validate(raw, model="test")
    assert res.questions[0].question_type == "unknown"


def test_malformed_json_rejected():
    with pytest.raises(ExtractionError):
        parse_and_validate("not json at all {{{", model="test")


def test_missing_questions_key_rejected():
    with pytest.raises(ExtractionError):
        parse_and_validate('{"answers": []}', model="test")


def test_missing_question_number_rejected():
    with pytest.raises(ExtractionError):
        parse_and_validate('{"questions": [{"question_text": "no number"}]}', model="test")


def test_markdown_fences_stripped():
    raw = '```json\n{"questions": [{"question_number": "1", "question_text": "Hi"}]}\n```'
    res = parse_and_validate(raw, model="test")
    assert len(res.questions) == 1


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "paper.gif"
    p.write_bytes(b"GIF89a....")
    with pytest.raises(ImageError):
        load_image_bytes(p)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ImageError):
        load_image_bytes(tmp_path / "nope.png")


def test_corrupt_image_rejected(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image at all")
    with pytest.raises(ImageError):
        load_image_bytes(p)


def test_oversized_image_rejected(tmp_path, monkeypatch):
    import paper_to_solution.config as cfg

    monkeypatch.setattr(cfg, "MAX_IMAGE_BYTES", 10)
    p = tmp_path / "a.png"
    img = Image.new("RGB", (50, 50), "white")
    img.save(p)
    with pytest.raises(ImageError):
        load_image_bytes(p)


def test_preprocess_downscales_huge_image():
    img = Image.new("RGB", (4000, 3000), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = preprocess_image(buf.getvalue())
    with Image.open(io.BytesIO(out)) as im:
        assert max(im.size) <= 2048


def test_preprocess_keeps_small_image_size():
    img = Image.new("RGB", (600, 400), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = preprocess_image(buf.getvalue())
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (600, 400)


def test_detect_mime_by_magic_bytes():
    assert detect_mime("x", b"\x89PNG\r\n\x1a\n....") == "image/png"
    assert detect_mime("x", b"\xff\xd8\xff....") == "image/jpeg"
