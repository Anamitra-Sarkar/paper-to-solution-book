"""Canonical question representation (team-wide contract).

The Question/Paper models below are kept field-for-field identical to the
canonical schema owned by the PDF ingestion component:

    yugal072/answer-book :: backend/app/models/loaders_models.py

verified identical on upstream branches main and backend (2026-09-19).

Sync procedure: if the upstream file changes, copy its Question/Paper
definitions here verbatim and update the converter + mapping table below.
Do NOT extend these models for OCR convenience; populate what vision can
populate and use the upstream conventions (None / defaults) otherwise.

Both ingestion paths converge here:
    PDF   -> PDF ingestion  -> Paper
    IMAGE -> Qwen vision    -> Paper  (via to_canonical_question)
Downstream code (LangGraph, answer generation) consumes Question dicts with
keys: number / section / text / marks / type / options / has_figure /
page / choice_group.
"""
from __future__ import annotations

import hashlib
import logging
import re as _re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical models (verbatim copy of upstream loaders_models.py, synced
# 2026-09-26; upstream added subject/class_name/board after the first sync)
# ---------------------------------------------------------------------------
class Question(BaseModel):
    number: str
    section: Optional[str] = None
    text: str
    marks: Optional[int] = None
    type: Literal['numerical', 'mcq', 'short', 'long'] = 'short'
    options: Optional[list[str]] = None
    has_figure: bool = False
    page: Optional[int] = None
    choice_group: Optional[str] = None


class Paper(BaseModel):
    paper_id: str
    fingerprint: str
    status: Literal['parsing', 'solving', 'ready', 'failed']
    subject: Optional[str] = None
    class_name: Optional[str] = None
    board: Optional[str] = None
    questions: List[Question] = Field(default_factory=list)
    total_questions: Optional[int] = None
    total_marks: Optional[int] = None
    sections: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Convergence: internal extraction detail -> canonical Question
# ---------------------------------------------------------------------------
# Internal classifier (vision / text parser) -> canonical type. Collapsed
# values are documented here; nothing is dropped silently: the full wording
# (including sub-parts and OR alternatives) always remains in Question.text.
TYPE_MAP = {
    "mcq": "mcq",
    "numerical": "numerical",
    "short_answer": "short",
    "descriptive": "long",
    "coding": "long",
    "fill_in_the_blank": "short",
    "true_false": "short",
    "unknown": "short",  # canonical default, same as upstream
}


def canonical_type(internal_type: str) -> str:
    return TYPE_MAP.get(internal_type, "short")


def to_canonical_question(q) -> Question:
    """Convert an internal ExtractedQuestion to the canonical Question.

    Accepts the internal model object (duck-typed) to avoid import cycles.
    Rules: options [] -> None; non-integral marks -> None (never fabricated);
    choice_group set to the question number when an OR alternative is present;
    confidence/extraction_notes are operational metadata, not part of the
    contract, and are logged instead of stored.
    """
    options = list(q.options) if q.options else None
    marks = None
    if q.marks is not None:
        try:
            f = float(q.marks)
            marks = int(f) if f.is_integer() else None
        except (TypeError, ValueError):
            marks = None
    text = q.question_text or ""
    choice_group = None
    for sub in q.subquestions or []:
        label = getattr(sub, "label", "") or ""
        if label.startswith("OR-"):
            choice_group = q.question_number
            break
    if choice_group is None and _re.search(r"\bOR\b", text):
        choice_group = q.question_number
    if q.extraction_notes or (q.confidence or 0) < 1.0:
        log.debug("Q%s confidence=%s notes=%s", q.question_number, q.confidence, q.extraction_notes)
    section = None
    if q.extraction_notes:
        m = _re.match(r"Section\s+([A-Z])\b", q.extraction_notes)
        if m:
            section = m.group(1)
    return Question(
        number=str(q.question_number),
        section=getattr(q, "section", None) or section,
        text=text,
        marks=marks,
        type=canonical_type(getattr(q, "question_type", "unknown")),
        options=options,
        has_figure=bool(getattr(q, "has_figure", False)),
        page=q.source_page,
        choice_group=choice_group,
    )


def fingerprint_bytes(*blobs: bytes) -> str:
    h = hashlib.sha256()
    for b in blobs:
        h.update(b)
    return h.hexdigest()


def _merge_metadata(*metas: dict) -> dict:
    """First non-null value wins per key (top tile / earlier page preferred)."""
    merged: dict = {}
    for meta in metas:
        for k, v in (meta or {}).items():
            if v and k not in merged:
                merged[k] = v
    return merged


def merge_metadata(*metas: dict) -> dict:
    """Public alias of the first-wins metadata merge."""
    return _merge_metadata(*metas)


def to_canonical_paper(
    questions: list[Question],
    source_bytes: list[bytes],
    metadata: dict | None = None,
    status: str = "ready",
) -> Paper:
    """Assemble a canonical Paper (same paper_id/fingerprint convention as PDF ingestion)."""
    fingerprint = fingerprint_bytes(*source_bytes) if source_bytes else fingerprint_bytes(b"")
    sections = list(dict.fromkeys(q.section for q in questions if q.section))
    return Paper(
        paper_id=f"pap_{fingerprint[:8]}",
        fingerprint=fingerprint,
        status=status,  # type: ignore[arg-type]
        subject=(metadata or {}).get("subject"),
        class_name=(metadata or {}).get("class"),
        board=(metadata or {}).get("board"),
        questions=questions,
        total_questions=len(questions),
        total_marks=sum(q.marks or 0 for q in questions) or None,
        sections=sections,
    )
