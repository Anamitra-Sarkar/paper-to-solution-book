# Paper to Solution Book — prototype

End-to-end: **Question paper (PDF/image) → question extraction (Groq `qwen/qwen3.8-27b` vision)
→ LangGraph (`START → solve → END`) → LLM answers.**

Team map: Anamitra = image input/OCR · Yugal = PDF ingestion · Yashwanth = LangGraph ·
Abhiram = answer generation · Likhitha = integration & testing.

## Setup

```bash
pip install -r requirements.txt   # or: pip install -e .
export GROQ_API_KEY="gsk_..."     # never hardcoded, never committed (.env is gitignored)
```

## Run

```bash
# Extract questions from image(s)
ptsb extract paper.png
ptsb extract page1.png page2.png
ptsb extract paper.pdf            # PDF pages -> same vision extractor

# Full pipeline: extract -> LangGraph -> answers
ptsb solve paper.png

# Demo web UI (upload image, inspect questions, get answers)
ptsb serve                       # -> http://localhost:8000
```

## API

- `extract_questions_from_image(path, page_number=1)` — image → `ExtractionResult`
- `extract_questions_from_images(paths)` — multi-page, `source_page` preserved, deduped
- `extract_questions_from_pdf(pdf_path)` — PDF → same extractor per rendered page
- `run_graph(questions)` — LangGraph `START → solve → END`
- `run_image_to_answers(paths)` / `run_pdf_to_answers(pdf)` — full pipeline

## Schema (per question)

`question_number · question_text · marks · question_type · options · subquestions ·
source_page · confidence · extraction_notes` — `null`/empty when unknown, never invented.

## Tests

```bash
pytest -m "not live"        # offline unit/integration (no key needed)
GROQ_API_KEY=... pytest -m live -v   # live Groq vision verification
```
