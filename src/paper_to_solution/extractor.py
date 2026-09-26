"""Image -> canonical Paper via Groq hosted Qwen vision.

Public surface returns the team-wide canonical schema (canonical.py,
identical to upstream models/loaders_models.py) so PDF and image ingestion
converge to one representation:

    IMAGE -> Qwen vision extraction -> canonical Paper -> shared downstream

    extract_questions_from_image(path, page_number=1) -> Paper
    extract_questions_from_images(paths)              -> Paper (multi-page)
    extract_questions_from_data_url(data_url, ...)    -> Paper
    extract_questions_from_pil(image, ...)            -> Paper (PDF-rendered pages)

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
from paper_to_solution.canonical import Paper, _merge_metadata, to_canonical_paper, to_canonical_question
from paper_to_solution.image_io import ImageError, preprocess_image, to_data_url
from paper_to_solution.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT
from paper_to_solution.schemas import ExtractedQuestion, ExtractionResult, SubQuestion

log = logging.getLogger(__name__)

VALID_QTYPES = {
    "mcq", "short_answer", "descriptive", "numerical",
    "coding", "fill_in_the_blank", "true_false", "unknown",
}


class ExtractionError(RuntimeError):
    pass


class TruncationError(ExtractionError):
    """Model output hit the token budget; partial data is never returned silently."""
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


def _parse_metadata(data: dict) -> dict:
    """Paper header metadata (subject/class/board); null-tolerant, never fabricated."""
    raw = data.get("metadata")
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key in ("subject", "class", "board"):
        val = raw.get(key)
        out[key] = str(val).strip() or None if val is not None else None
    return {k: v for k, v in out.items() if v}


def parse_and_validate(raw_text: str, model: str, default_page: int = 1) -> ExtractionResult:
    """Parse model JSON -> internal ExtractionResult. Raises ExtractionError on malformed output."""
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
        qd.setdefault("section", None)
        qd.setdefault("has_figure", False)
        qd.setdefault("choice_group", None)
        if "question_number" not in qd or qd["question_number"] in (None, ""):
            raise ExtractionError(f"questions[{i}] missing required 'question_number'")
        qd["question_number"] = str(qd["question_number"])
        if qd.get("question_type") not in VALID_QTYPES:
            qd["question_type"] = "unknown"
        if qd.get("options") is None:
            qd["options"] = []
        if qd.get("subquestions") is None:
            qd["subquestions"] = []
        if qd.get("section") is not None:
            sec = str(qd["section"]).strip().upper().replace("SECTION ", "")
            qd["section"] = sec or None
        qd["has_figure"] = bool(qd.get("has_figure", False))
        if qd.get("choice_group") is not None:
            qd["choice_group"] = str(qd["choice_group"])
        elif " OR:" in f" {qd.get('question_text', '')}":
            # Fallback: internal choice visible in text but not flagged.
            qd["choice_group"] = qd["question_number"]
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

    return ExtractionResult(questions=questions, raw_response=raw_text, model=model,
                              metadata=_parse_metadata(data))


def _extract_internal(data_url: str, source_page: int, model: str) -> ExtractionResult:
    """Single vision call for one already-prepared image data URL (internal step)."""
    client = _client()
    last_err: Exception | None = None
    for attempt in (1, 2):  # single retry on transport errors only
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                # Verified 2026-09-19: on the on_demand tier (OTPM limit 1000)
                # larger per-request budgets are rejected outright, so the
                # cap pairs with small render sizes (see config). Pages that
                # still overflow are handled by tile-splitting below.
                # Truncation is surfaced, never silent.
                max_tokens=config.VISION_MAX_TOKENS,
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
            wait = 10 * attempt if isinstance(e, RateLimitError) else 2 * attempt
            log.warning("Groq transient error (attempt %d): %s; waiting %ss",
                        attempt, type(e).__name__, wait)
            time.sleep(wait)
    else:
        raise ExtractionError(f"Groq API transport failure after retry: {last_err}")
    try:
        content = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
    except (APIStatusError,) as e:
        raise ExtractionError(f"Groq API error: {e}") from e
    if not content.strip():
        raise ExtractionError("Model returned an empty response")
    if finish == "length":
        raise TruncationError(
            "Model output hit the token budget before all questions were emitted."
        )
    result = parse_and_validate(content, model=model, default_page=source_page)
    return result


def _finalize(result: ExtractionResult, source_page: int) -> ExtractionResult:
    """Stamp page + provenance (applied once per page, after any tile merge)."""
    for q in result.questions:
        if not q.source_page:
            q.source_page = source_page
        if q.extraction_notes:
            q.extraction_notes += " | Source: vision OCR (Qwen via Groq)"
        else:
            q.extraction_notes = "Source: vision OCR (Qwen via Groq)"
    return result


def _pil_to_data_url(image) -> str:
    import io as _io

    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    return to_data_url(preprocess_image(buf.getvalue()))


def _data_url_to_pil(data_url: str):
    import base64 as _b64
    import io as _io

    from PIL import Image as _Image

    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    return _Image.open(_io.BytesIO(_b64.b64decode(payload))).convert("RGB")


TILE_OVERLAP = 0.25
MIN_TILE_HEIGHT = 800


def _merge_tile_questions(top: list[ExtractedQuestion], bottom: list[ExtractedQuestion]):
    """Merge two same-page tile extractions without losing boundary content.

    - Identical repeats (overlap region) are deduped, keeping the union of options.
    - A question straddling the split appears partially in both tiles and is
      reassembled in reading order, flagged in extraction_notes.
    - Options are unioned order-preservingly in every case, so an option seen
      by either tile always survives. Text is only concatenated when the two
      fragments are genuinely disjoint; never blindly.
    """
    merged: list[ExtractedQuestion] = []
    seen: set[tuple[str, str, tuple]] = set()
    by_number: dict[str, ExtractedQuestion] = {}
    for q in list(top) + list(bottom):
        key = (q.question_number.strip(), q.question_text.strip()[:200], tuple(q.options))
        if key in seen:
            continue
        seen.add(key)
        prev = by_number.get(q.question_number.strip())
        if prev is not None:
            for o in q.options:
                if o not in prev.options:
                    prev.options.append(o)
            if prev.question_text != q.question_text:
                if q.question_text in prev.question_text or prev.question_text in q.question_text:
                    prev.question_text = max(prev.question_text, q.question_text, key=len)
                else:
                    prev.question_text = f"{prev.question_text} {q.question_text}".strip()
                    note = "Reassembled from overlapping tiles."
                    prev.extraction_notes = (
                        f"{prev.extraction_notes} | {note}" if prev.extraction_notes else note
                    )
            if prev.marks is None:
                prev.marks = q.marks
            prev.confidence = min(prev.confidence, q.confidence)
            if not prev.section and q.section:
                prev.section = q.section
            prev.has_figure = prev.has_figure or q.has_figure
            continue
        by_number[q.question_number.strip()] = q
        merged.append(q)
    return merged


def _find_split_row(image, search_radius: int = 120) -> int:
    """Row index near the vertical middle that cuts through whitespace, not text.

    Uses the horizontal projection profile: rows darker than near-white count
    as ink. Returns the quietest row closest to the middle within the search
    band, so a crop edge never slices a text line (e.g. an MCQ option) in two.
    Falls back to the exact middle when no whitespace band is found.
    """
    import statistics as _stats

    gray = image.convert("L")
    w, h = gray.size
    px = gray.load()
    darkness = []
    for y in range(h):
        row = (px[x, y] for x in range(0, w, 4))
        darkness.append(sum(255 - v for v in row))
    mid = h // 2
    lo, hi = max(0, mid - search_radius), min(h - 1, mid + search_radius)
    band = darkness[lo:hi + 1]
    if not band:
        return mid
    threshold = max(_stats.mean(band) * 0.25, 1.0)
    best, best_key = mid, None
    for i, d in enumerate(band):
        y = lo + i
        key = (d > threshold, abs(y - mid), -y)
        if best_key is None or key < best_key:
            best, best_key = y, key
    return best


def _extract_tiles(image, source_page: int, model: str, _depth: int = 0) -> ExtractionResult:
    w, h = image.size
    overlap = int(h * TILE_OVERLAP)
    split = _find_split_row(image)
    tiles = [image.crop((0, 0, w, split + overlap)), image.crop((0, split - overlap, w, h))]
    return _extract_tile_list(tiles, source_page, model, _depth)


def _extract_tile_list(tiles, source_page: int, model: str, _depth: int) -> ExtractionResult:
    merged_qs: list[ExtractedQuestion] = []
    tile_metas: list[dict] = []
    raw_parts: list[str] = []
    for tile in tiles:
        try:
            tile_result = _extract_internal(_pil_to_data_url(tile), source_page, model)
        except TruncationError:
            if _depth >= 1 or tile.size[1] < MIN_TILE_HEIGHT:
                raise
            log.info("Tile overflowed; splitting again (depth %d).", _depth + 1)
            w, h = tile.size
            overlap = int(h * TILE_OVERLAP)
            split = _find_split_row(tile)
            sub = [tile.crop((0, 0, w, split + overlap)),
                   tile.crop((0, split - overlap, w, h))]
            tile_result = _extract_tile_list(sub, source_page, model, _depth + 1)
        raw_parts.append(tile_result.raw_response or "")
        tile_metas.append(tile_result.metadata)
        merged_qs = _merge_tile_questions(merged_qs, tile_result.questions)
    return ExtractionResult(questions=merged_qs, raw_response="\n".join(raw_parts), model=model,
                            metadata=_merge_metadata(*tile_metas))


def _extract_page(image, source_page: int, model: str) -> ExtractionResult:
    """Extract one page; tile-split with overlap only if the token budget overflows."""
    try:
        return _finalize(_extract_internal(_pil_to_data_url(image), source_page, model), source_page)
    except TruncationError:
        if image.size[1] < MIN_TILE_HEIGHT:
            raise
        log.info("Page %d overflowed the token budget; retrying as overlapping tiles.", source_page)
        return _finalize(_extract_tiles(image, source_page, model), source_page)


def extract_questions_from_data_url(
    data_url: str,
    source_page: int = 1,
    model: str | None = None,
) -> Paper:
    """Single image (as data URL) -> canonical Paper, tiling if the budget overflows."""
    model = model or config.EXTRACTION_MODEL
    try:
        image = _data_url_to_pil(data_url)
        result = _extract_page(image, source_page, model)
    except ExtractionError:
        raise
    except Exception as e:  # unexpected errors -> clean message, no secrets
        raise ExtractionError(f"Groq API failure: {type(e).__name__}: {e}") from e
    return to_canonical_paper(
        [to_canonical_question(q) for q in result.questions],
        [data_url.encode("utf-8")],
        result.metadata,
    )


def extract_questions_from_image(
    path: str | Path,
    page_number: int = 1,
    model: str | None = None,
) -> Paper:
    """image -> canonical Paper (single page; tiles itself if dense)."""
    from paper_to_solution.image_io import load_image_bytes

    from PIL import Image as _Image
    import io as _io

    try:
        raw = load_image_bytes(path)[0]
        image = _Image.open(_io.BytesIO(raw)).convert("RGB")
    except ImageError:
        raise
    try:
        result = _extract_page(image, page_number, model or config.EXTRACTION_MODEL)
    except ExtractionError:
        raise
    except Exception as e:  # unexpected errors -> clean message, no secrets
        raise ExtractionError(f"Groq API failure: {type(e).__name__}: {e}") from e
    for q in result.questions:
        q.source_page = page_number
    return to_canonical_paper([to_canonical_question(q) for q in result.questions], [raw],
                              result.metadata)


def extract_questions_from_images(
    paths: list[str | Path],
    model: str | None = None,
) -> Paper:
    """Multi-page: one vision call per page -> single canonical Paper.

    - Preserves page per question.
    - Dedupes exact repeats across page boundaries (same number + same text).
    - Flags likely continuations via logged notes (never merges silently).
    - Dense pages tile themselves (see _extract_page).
    """
    from paper_to_solution.image_io import load_image_bytes

    from PIL import Image as _Image
    import io as _io

    model = model or config.EXTRACTION_MODEL
    merged: list[ExtractedQuestion] = []
    seen: set[tuple[str, str]] = set()
    sources: list[bytes] = []
    metas: list[dict] = []
    for idx, p in enumerate(paths, start=1):
        raw = load_image_bytes(p)[0]
        sources.append(raw)
        image = _Image.open(_io.BytesIO(raw)).convert("RGB")
        result = _extract_page(image, idx, model)
        metas.append(result.metadata)
        for q in result.questions:
            q.source_page = idx
            key = (q.question_number.strip(), q.question_text.strip()[:200])
            if key in seen:
                continue  # exact duplicate across a page boundary
            seen.add(key)
            if any(m.question_number == q.question_number and m.question_text != q.question_text for m in merged):
                log.info(
                    "Q%s on page %d possibly continues from previous page; kept separate.",
                    q.question_number, idx,
                )
            merged.append(q)
    return to_canonical_paper([to_canonical_question(q) for q in merged], sources,
                              _merge_metadata(*metas))


def extract_questions_from_pil(image, source_page: int = 1, model: str | None = None) -> Paper:
    """Entry point for PDF-rendered pages: pass a PIL image directly -> canonical Paper."""
    import io as _io

    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    png = buf.getvalue()
    result = _extract_page(image.convert("RGB"), source_page, model or config.EXTRACTION_MODEL)
    for q in result.questions:
        q.source_page = source_page
    return to_canonical_paper([to_canonical_question(q) for q in result.questions], [png],
                              result.metadata)
