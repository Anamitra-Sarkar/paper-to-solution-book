"""Deterministic parser: digital PDF text layer -> structured questions.

Used by the hybrid PDF ingestion path. When a page carries a real text layer
(as opposed to a scanned/image-only page), parsing it directly is faster,
cheaper and more faithful than re-reading it through vision. Pages without a
usable text layer are routed to the vision extractor instead (see pdf_ingest).

Expected layout (AparExam papers and similar tabular papers)::

    <question-number>
    <question text, one or more lines>
    [(A) ... (B) ... (C) ... (D) ...]      # MCQ options, one per line
    <marks>                                 # standalone number
    [[Ch ...: ...]]                         # optional chapter tag
    [OR                                     # optional alternative question
     <alternative text, no number/marks>]

A question may continue on the next page (chapter tag and/or OR block appear
before the next question number). Continuations are merged into the open
question, never emitted as separate entries.
"""
from __future__ import annotations

import re

from paper_to_solution.schemas import ExtractedQuestion

_QNO_RE = re.compile(r"^\d{1,3}$")
_MARKS_RE = re.compile(r"^\d{1,3}$")
_CHAPTER_RE = re.compile(r"^\[Ch\b.*\]$")
_OPTION_RE = re.compile(r"^\(([A-D])\)\s*(.*)$")
_OPTION_ALT_RE = re.compile(r"^([A-D])[).]\s+(.*)$")
_SECTION_RE = re.compile(r"^Section\s+[A-Z]$")
_ATTEMPT_RE = re.compile(r"^Attempt\s+")
_ROMAN_SPLIT_RE = re.compile(r"\((i{1,3}|iv|v|vi)\)")

_SKIP_PREFIXES = ("aparexam.com", "Page ", "Q.NO.", "QUESTIONS", "MARKS")
_SKIP_CONTAINS = ("Subject:", "Total Marks:", "Time Allowed:", "Class:")


def _is_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith(_SKIP_PREFIXES):
        return True
    if any(k in s for k in _SKIP_CONTAINS):
        return True
    if _SECTION_RE.match(s) or _ATTEMPT_RE.match(s):
        return True
    if s == "General Instructions:":
        return True
    if re.match(r"^\d+\.\s+(Read|All|Marks|Write)\b", s):
        return True
    if "Annual Exam" in s and "Marks" in s:
        return True
    return False


def _classify(text: str, marks, has_options: bool) -> str:
    if has_options:
        return "mcq"
    low = text.lower()
    if re.search(r"\b(solve|calculate|compute|find|determine|verify|expand|simplify|evaluate)\b", low):
        return "numerical"
    if re.search(r"\b(prove|explain|describe|discuss|justify|elaborate)\b", low):
        return "descriptive"
    try:
        m = float(marks) if marks is not None else 0
    except (TypeError, ValueError):
        m = 0
    if m >= 4:
        return "descriptive"
    if m >= 1:
        return "short_answer"
    return "unknown"


def _split_subquestions(text: str) -> list[tuple[str, str]]:
    """Split '(i) ... (ii) ...' style parts. Non-destructive: returns [] unless clean."""
    parts = _ROMAN_SPLIT_RE.split(text)
    if len(parts) < 3:
        return []
    subs: list[tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        label, body = parts[i].strip(), parts[i + 1].strip()
        if len(body) < 3:
            return []
        subs.append((label, body))
    return subs if len(subs) >= 2 else []


def parse_text_pages(pages: list[tuple[int, str]]) -> list[ExtractedQuestion]:
    """Parse (page_number, page_text) pairs into structured questions.

    Cross-page continuations (a page that opens with a chapter tag / OR block
    instead of a question number) are merged into the previously open question.
    """
    questions: list[ExtractedQuestion] = []
    current: dict | None = None
    section = ""
    in_or_block = False

    def flush() -> None:
        nonlocal current, in_or_block
        if current and current["text_lines"]:
            questions.append(_build(current))
        current = None
        in_or_block = False

    def ensure_continuation() -> None:
        # Content without a question number belongs to the previous question.
        nonlocal current
        if current is None and questions:
            prev = questions.pop()
            current = {
                "number": prev.question_number,
                "text_lines": [prev.question_text],
                "options": list(prev.options),
                "marks": prev.marks,
                "notes": [prev.extraction_notes] if prev.extraction_notes else [],
                "page": prev.source_page,
                "subs": [(s.label, s.text) for s in prev.subquestions],
                "qtype": prev.question_type,
                "saw_marks": prev.marks is not None,
                "saw_chapter": False,
                "section": (prev.extraction_notes or "").split(" | ")[0]
                if (prev.extraction_notes or "").startswith("Section") else "",
            }

    for page_no, raw_text in pages:
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _SECTION_RE.match(line):
                section = line
                continue
            if _is_header(line):
                continue

            if _QNO_RE.match(line) and not in_or_block and (
                current is None or current.get("saw_marks") or current.get("saw_chapter")
                or not current["text_lines"]
            ):
                # A new question starts only after the previous one closed
                # (marks/chapter seen) or when nothing is open.
                if current and current["text_lines"]:
                    flush()
                current = {"number": line, "text_lines": [], "options": [],
                           "marks": None, "notes": [], "page": page_no,
                           "subs": [], "qtype": "unknown",
                           "saw_marks": False, "saw_chapter": False,
                           "section": section}
                continue
            if line == "OR":
                ensure_continuation()
                if current is None:
                    continue
                current["text_lines"].append("OR")
                in_or_block = True
                continue

            if _CHAPTER_RE.match(line):
                ensure_continuation()
                if current is None:
                    continue
                tag = line.strip("[]")
                current["notes"].append(tag)
                current["saw_chapter"] = True
                in_or_block = False
                continue

            m = _OPTION_RE.match(line) or _OPTION_ALT_RE.match(line)
            if m and current is not None and not in_or_block:
                current["options"].append(f"({m.group(1)}) {m.group(2).strip()}")
                continue

            if (
                _MARKS_RE.match(line)
                and current is not None
                and current["text_lines"]
                and not in_or_block
                and not current.get("saw_marks")
                and _looks_like_marks(line)
            ):
                current["marks"] = _to_number(line)
                current["saw_marks"] = True
                in_or_block = False
                continue

            # Ordinary content line.
            if current is None:
                ensure_continuation()
                if current is None:
                    continue
            current["text_lines"].append(line)

    flush()
    return questions


def _looks_like_marks(line: str) -> bool:
    """A standalone number is marks only if it closes the question body."""
    try:
        v = int(line)
    except ValueError:
        return False
    return 1 <= v <= 20


def _to_number(s: str):
    try:
        return int(s)
    except ValueError:
        return None


def _build(current: dict) -> ExtractedQuestion:
    text = " ".join(current["text_lines"]).strip()
    text = re.sub(r"\s+", " ", text)
    options = current["options"]
    # Split sub-parts separately for the main question and the OR alternative
    # so the two (i)-(iv) sequences never merge into one list.
    main_text, sep, or_text = text.partition(" OR ")
    subs = [(label, body) for label, body in _split_subquestions(main_text)]
    if sep and or_text.strip():
        subs += [(f"OR-{label}", body) for label, body in _split_subquestions(or_text)]
    qtype = _classify(text, current["marks"], bool(options))
    notes = [n for n in current["notes"] if n]
    if current.get("section"):
        notes.insert(0, current["section"])
    notes.append("Source: digital text layer (no vision used)")
    has_figure = bool(
        re.search(
            r"\b(figure|diagram|graph|plot|shown below|shown above|given figure)\b",
            text,
            re.IGNORECASE,
        )
    )
    return ExtractedQuestion(
        question_number=current["number"],
        question_text=text,
        marks=current["marks"],
        question_type=qtype,  # type: ignore[arg-type]
        options=options,
        subquestions=[{"label": lb, "text": bd, "marks": None} for lb, bd in subs],  # type: ignore[list-item]
        source_page=current["page"],
        confidence=0.99,
        extraction_notes=" | ".join(notes) if notes else None,
        section=(current.get("section") or "").replace("Section ", "") or None,
        has_figure=has_figure,
    )
