"""Image -> extracted structured questions via Groq hosted Qwen vision (Anamitra).

Public surface (also used by Yugal's PDF pipeline and Yashwanth's LangGraph):
    extract_questions_from_image(path, page_number=1) -> ExtractionResult
    extract_questions_from_images(paths)              -> ExtractionResult (multi-page)
    extract_questions_from_data_url(data_url, ...)    -> ExtractionResult (PDF-rendered pages)

One vision call per image (no duplicate OCR calls). Retries once on transport
errors; malformed JSON -> safe recovery path, never fake data.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError

from paper_to_solution import config
from paper_to_solution.image_io import ImageError, prepare_image_for_model, preprocess_image, to_data_url
from paper_to_solution.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT
from paper_to_solution.schemas import ExtractedQuestion, ExtractionResult, SubQuestion

log = logging.getLogger(__name__)

VALID_QTYPES = {
    "mcq", "short_answer", "descriptive", "numerical",
    "coding", "fill_in_the_blank", "true_false", "unknown",
}


class ExtractionError(RuntimeError):
    pass


def _client() -> Groq:
    return Groq(api_key=config.get_groq_api_key(), timeout=config.GROQ_TIMEOUT_S)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rsplit("```", 1)[0].strip():
            t = t.rsplit("```", 1)[0]
    return t.strip()


def parse_and_validate(raw_text: str, model: str, default_page: int = 1) -> ExtractionResult:
    """Parse model JSON -> ExtractionResult. Raises ExtractionError on malformed output."""
    cleaned = _strip_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model did not return valid JSON: {e}") from e
    if not isinstance(data, dict) or "questions" not in data:
        raise ExtractionError("Model response missing required top-level 'questions' key")
    if not isinstance(data["questions"], list):
        raise ExtractionError("'questions' must be a list")

    questions: list[ExtractedQuestion] = []
    for i, item in enumerate(data["questions"]):
        if not isinstance(item, dict):
            raise ExtractionError(f"questions[{i}] is not an object")
        qd = dict(item)
        # Normalize missing optional fields; never fabricate.
        qd.setdefault("question_text", "")
        qd.setdefault("marks", None)
        qd.setdefault("options", [])
        qd.setdefault("subquestions", [])
        qd.setdefault("source_page", default_page)
        qd.setdefault("confidence", 0.0)
        qd.setdefault("extraction_notes", None)
        if "question_number" not in qd or qd["question_number"] in (None, ""):
            raise ExtractionError(f"questions[{i}] missing required 'question_number'")
        qd["question_number"] = str(qd["question_number"])
        if qd.get("question_type") not in VALID_QTYPES:
            qd["question_type"] = "unknown"
        if qd.get("options") is None:
            qd["options"] = []
        if qd.get("subquestions") is None:
            qd["subquestions"] = []
        norm_subs = []
        for s in qd["subquestions"]:
            if isinstance(s, dict):
                norm_subs.append(
                    SubQuestion(
                        label=str(s.get("label", "")),
                        text=str(s.get("text", "")),
                        marks=s.get("marks"),
                    )
                )
            elif isinstance(s, str):
                norm_subs.append(SubQuestion(label="", text=s))
        qd["subquestions"] = norm_subs
        try:
            questions.append(ExtractedQuestion(**qd))
        except Exception as e:
            raise ExtractionError(f"questions[{i}] failed schema validation: {e}") from e

    return ExtractionResult(questions=questions, raw_response=raw_text, model=model)


def extract_questions_from_data_url(
    data_url: str,
    source_page: int = 1,
    model: str | None = None,
) -> ExtractionResult:
    """Single vision call for one already-prepared image data URL."""
    model = model or config.EXTRACTION_MODEL
    client = _client()
    last_err: Exception | None = None
    for attempt in (1, 2):  # single retry on transport errors only
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            break
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e
            log.warning("Groq transient error (attempt %d): %s", attempt, type(e).__name__)
            time.sleep(2 * attempt)
    else:
        raise ExtractionError(f"Groq API transport failure after retry: {last_err}")
    try:
        content = resp.choices[0].message.content or ""
    except (APIStatusError,) as e:
        raise ExtractionError(f"Groq API error: {e}") from e
    if not content.strip():
        raise ExtractionError("Model returned an empty response")
    result = parse_and_validate(content, model=model, default_page=source_page)
    # Stamp the actual page (model may omit it)
    for q in result.questions:
        if not q.source_page:
            q.source_page = source_page
    return result


def extract_questions_from_image(
    path: str | Path,
    page_number: int = 1,
    model: str | None = None,
) -> ExtractionResult:
    """image -> structured questions (single page)."""
    try:
        data_url, _meta = prepare_image_for_model(path)
    except ImageError:
        raise
    try:
        result = extract_questions_from_data_url(data_url, source_page=page_number, model=model)
    except ExtractionError:
        raise
    except Exception as e:  # Groq SDK errors -> clean actionable message, no secrets
        raise ExtractionError(f"Groq API failure: {type(e).__name__}: {e}") from e
    for q in result.questions:
        q.source_page = page_number
    return result


def extract_questions_from_images(
    paths: list[str | Path],
    model: str | None = None,
) -> ExtractionResult:
    """Multi-page: one vision call per page (sequential = no rate-limit spikes).

    - Preserves source_page per question.
    - Dedupes exact repeats across page boundaries (same number + same text).
    - Flags likely continuations via extraction_notes (never merges silently).
    """
    model = model or config.EXTRACTION_MODEL
    merged: list[ExtractedQuestion] = []
    seen: set[tuple[str, str]] = set()
    raw_parts: list[str] = []
    for idx, p in enumerate(paths, start=1):
        result = extract_questions_from_image(p, page_number=idx, model=model)
        raw_parts.append(result.raw_response or "")
        for q in result.questions:
            key = (q.question_number.strip(), q.question_text.strip()[:200])
            if key in seen:
                continue  # exact duplicate across a page boundary
            seen.add(key)
            # Continuation hint: same question number already seen with different text
            if any(m.question_number == q.question_number and m.question_text != q.question_text for m in merged):
                note = f"Possibly continues '{q.question_number}' from page {idx - 1}; kept as separate entry."
                q.extraction_notes = f"{q.extraction_notes} | {note}" if q.extraction_notes else note
            merged.append(q)
    return ExtractionResult(questions=merged, raw_response="\n".join(raw_parts), model=model)


def extract_questions_from_pil(image, source_page: int = 1, model: str | None = None) -> ExtractionResult:
    """Entry point for PDF-rendered pages (Yugal): pass a PIL image directly."""
    import io as _io

    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    data_url = to_data_url(preprocess_image(buf.getvalue()))
    return extract_questions_from_data_url(data_url, source_page=source_page, model=model)
