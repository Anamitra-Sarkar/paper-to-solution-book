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

## Question schema (canonical, team-wide)

Both ingestion paths return the same `Paper`/`Question` representation,
field-for-field identical to the canonical `models/loaders_models.py`
contract owned by the PDF ingestion component. Downstream code never needs
to know whether a question came from a PDF or an image.

Per question: `number · section · text · marks · type · options ·
has_figure · page · choice_group`, where `type` is one of
`numerical, mcq, short, long`. Unknown or invisible fields use `null`,
`false`, or the upstream defaults — values are never invented.

The vision/text classifiers use a finer internal vocabulary that is mapped
at the boundary (documented in `canonical.py`):

| internal | canonical |
|---|---|
| mcq | mcq |
| numerical | numerical |
| short_answer, fill_in_the_blank, true_false, unknown | short |
| descriptive, coding | long |

The full wording — including sub-parts and OR alternatives — always remains
in `text`; questions offering an internal choice carry
`choice_group` set to their number, matching PDF ingestion behavior.

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

To verify any paper from the shared set exactly as in the acceptance test,
see `docs/TESTING_GUIDE.md` and run `scripts/test_random_paper.py`.

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
- Platform limit: the Groq `on_demand` tier enforces OTPM 1000 and rejects
  oversized single vision requests. Rendered PDF pages are rasterized at
  100dpi (`VISION_RENDER_DPI`) under a 1000-token output budget
  (`VISION_MAX_TOKENS`, both overridable via environment). A page that
  still overflows is automatically retried as two overlapping tiles whose
  extractions are merged with overlap-dedup and straddler reassembly.
  Budget overruns that cannot be tiled fail loudly instead of truncating
  silently. The free tier also caps daily tokens (TPD 200000); sustained
  bursts may need to wait for the quota window.

## Demo checklist

1. Upload a question-paper image or PDF.
2. Extraction runs (vision via Groq and/or text parser).
3. Structured questions render with number, text, marks, type,
   options/subquestions, page and confidence.
4. Each question passes through LangGraph `solve`.
5. Answers render beneath their questions.
