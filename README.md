# Paper to Solution Book

End-to-end prototype that converts a question paper (PDF or image) into
structured questions and produces model-generated answers:

```
Question paper (PDF / PNG / JPEG / WebP)
  -> Question extraction (digital text layer and/or Qwen vision via Groq)
  -> Structured questions
  -> LangGraph pipeline (START -> solve -> END)
  -> Answers
```

## Architecture

```
                        +------------------+
                        |  Question paper  |
                        |   PDF / image    |
                        +--------+---------+
                                 |
                    +------------v-------------+
                    |      PDF ingestion       |  pdf_ingest.py
                    |  per-page routing:       |
                    |  text layer -> parser    |  text_parser.py
                    |  image-only -> vision    |  extractor.py (image_io.py)
                    +------------+-------------+
                                 | ExtractionResult
                    +------------v-------------+
                    |  LangGraph (START ->     |  graph.py
                    |   solve -> END)          |
                    +------------+-------------+
                                 | SolvedQuestion[]
                    +------------v-------------+
                    |  Answer generation       |  answer.py
                    |  (text LLM via Groq)     |
                    +--------------------------+
```

Direct image uploads skip PDF ingestion and go straight to the vision
extractor. PDF pages rendered to images reuse the exact same extractor, so
there is a single vision path regardless of input origin.

## Module responsibilities

| Module | Owner | Responsibility |
|---|---|---|
| `image_io.py`, `extractor.py`, `prompts.py` | Image input / OCR | Image validation, lightweight preprocessing, Qwen vision extraction, response validation |
| `pdf_ingest.py`, `text_parser.py` | PDF ingestion | Digital-text parsing with vision fallback per page |
| `graph.py`, `schemas.py` (state) | LangGraph | `START -> solve -> END` over extracted questions |
| `answer.py` | Answer generation | `ExtractedQuestion -> SolvedQuestion` via text LLM |
| `pipeline.py`, `cli.py`, `app.py` | Integration | End-to-end helpers, CLI, demo web UI and JSON API |

## Question schema

Every extracted question carries the same fields, regardless of source:

`question_number · question_text · marks · question_type · options ·
subquestions · source_page · confidence · extraction_notes`

Unknown fields use `null` (or empty lists); values are never invented.
`question_type` is one of `mcq, short_answer, descriptive, numerical,
coding, fill_in_the_blank, true_false, unknown`. `extraction_notes` always
records provenance (`Source: digital text layer ...` or
`Source: vision OCR ...`) plus chapter/section tags when available.

## PDF ingestion modes

`extract_questions_from_pdf(path, mode=...)`:

- `auto` (default) — pages with a usable text layer are parsed
  deterministically; scanned/image-only pages fall back to vision.
- `vision` — every page goes through vision (picks up diagrams even when
  text exists).
- `text` — text layer only; fully offline, no API calls.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."     # required for vision and answers
export EXTRACTION_MODEL="qwen/qwen3.8-27b"        # default
export ANSWER_MODEL="openai/gpt-oss-120b"         # default
```

Secrets are read from the environment only (see `.env.example`); nothing is
hardcoded or committed.

## Usage

```bash
PYTHONPATH=src python3 -m paper_to_solution.cli extract paper.png
PYTHONPATH=src python3 -m paper_to_solution.cli extract paper.pdf --mode auto
PYTHONPATH=src python3 -m paper_to_solution.cli solve paper.pdf
PYTHONPATH=src python3 -m paper_to_solution.cli serve   # demo UI on :8000
```

Python API:

```python
from paper_to_solution.extractor import extract_questions_from_image
from paper_to_solution.pdf_ingest import extract_questions_from_pdf
from paper_to_solution.pipeline import run_image_to_answers, run_pdf_to_answers

result = extract_questions_from_image("paper.png")          # single image
result = extract_questions_from_pdf("paper.pdf")            # hybrid PDF
out = run_pdf_to_answers("paper.pdf")                       # full pipeline
```

## Testing

```bash
pytest -m "not live"                  # offline suite, no credentials needed
GROQ_API_KEY=... pytest -m live -v    # live Groq verification
```

The offline suite covers schema validation, malformed-model-response
handling, image validation, multi-page merge/dedup, LangGraph wiring, and
the text parser against the vendored testcase paper
(`tests/assets/question_paper_455.pdf`, CBSE Class 9 Mathematics, 9 pages,
from the answer-book `test_papers` set).

## Reference results (testcase paper)

- `mode="text"`: 37/37 questions, correct order, no missing marks, MCQ
  options, OR alternatives, cross-page continuations and (i)-(iv) sub-parts
  preserved.
- Image-only rendering of a real page through vision: Q4-Q8 recovered with
  correct numbers, marks, options and Unicode content.
- Known fidelity limits: combining marks such as the overline in `0.45̄`
  and subscript styling may be simplified (`sₙ` -> `s_n`); illegible input
  can still yield overconfident output, so `confidence` should not be the
  sole quality signal for poor scans.

## Demo checklist

1. Upload a question-paper image or PDF.
2. Extraction runs (vision via Groq and/or text parser).
3. Structured questions render with number, text, marks, type,
   options/subquestions, page and confidence.
4. Each question passes through LangGraph `solve`.
5. Answers render beneath their questions.
