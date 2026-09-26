# Paper Test Guide

Procedure for verifying extraction and answer quality on any paper from the
shared `test_papers` set. Takes about five minutes, most of it API time.

## Prerequisites

```bash
git clone https://github.com/Anamitra-Sarkar/paper-to-solution-book
cd paper-to-solution-book
pip install -r requirements.txt   # use a virtualenv first if your machine
                                  # already has conflicting packages:
                                  # python3 -m venv .venv && source .venv/bin/activate
export GROQ_API_KEY="gsk_..."   # only needed for the answer sample
```

## Run the verification

```bash
# Specific paper:
python scripts/test_random_paper.py --paper 464

# Random paper (seeded, so the pick is reproducible):
python scripts/test_random_paper.py --random --seed 7

# Identification only, no API calls:
python scripts/test_random_paper.py --paper 464 --no-solve

# Larger answer sample, plus machine-readable output:
python scripts/test_random_paper.py --paper 464 --sample 6 --json-out report.json
```

## Reading the report

The script prints three sections:

1. **Identification** — question count, numbering contiguity (expects
   1..N with no gaps), marks presence, page coverage, type distribution,
   and MCQ option presence. Any problem is listed explicitly and the
   script exits non-zero.
2. **Answers** — a sample spread across question types (MCQs plus
   descriptive/numerical), each printed with its full answer text.
3. **Verdict** — `PASS` means identification is structurally clean.
   Answer *correctness* is always graded by eye from the printed sample:
   check option letters on MCQs, check numbers and working on numerical
   questions, and check factual claims on descriptive ones.

## What good looks like

- Count matches the paper's numbering; sequence is contiguous.
- Every question has marks; every MCQ has its options verbatim.
- OR alternatives and case-study sub-parts appear inside their parent
  question, not as separate entries.
- Answers select the right MCQ options and show valid working.

## Known fidelity limits

Vision output may simplify typographic details (combining overlines,
subscript styling such as `sₙ` rendered as `s_n`). Mathematical notation
should be proofread before any result is treated as final. Low-quality
scans deserve manual review regardless of the reported confidence value.

## Bad-phone-photo testing (Week 3)

Build deterministic photo fixtures from a real paper, then grade extraction:

```bash
python scripts/make_bad_photos.py --pdf tests/assets/question_paper_455.pdf \
    --pages 2 6 --out test_runs/bad_phones --seed 3
GROQ_API_KEY=... python scripts/test_phone_photos.py \
    --manifest test_runs/bad_phones/manifest.json
```

Grade one photo at a time to stay inside the Groq free-tier rate limits:

```bash
GROQ_API_KEY=... python scripts/test_phone_photos.py \
    --manifest test_runs/bad_phones/manifest.json --sample 1
```

Increase `--sample` one step at a time. Each photo is graded PASS only on
exact question count plus exact per-question marks against text-layer ground
truth. Fixtures are synthetic degradations of real pages (declared in the
manifest), used until real Set A phone photos are available.
