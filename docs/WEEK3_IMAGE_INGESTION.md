# Week 3 — Image / Photo Ingestion

## Objective

Make image/photo ingestion reliably extract questions, marks, sections,
figures, choices, subject, class, and board from real phone photos, and keep
the output compatible with the canonical LangGraph `Question`/`Paper`
structure shared with PDF ingestion.

## Starting point (Week 1 + Week 2, reused as-is)

- Groq-hosted Qwen vision extraction (`qwen/qwen3.8-27b`), one call per image.
- Strict extraction prompt with no-invention rules and JSON validation.
- Canonical `Paper`/`Question` output identical to `models/loaders_models.py`.
- Hybrid PDF ingestion (text layer + vision fallback) converging on the same schema.
- Tile-split fallback for token-budget overflow with overlap dedup and
  whitespace-snapped splits.
- LangGraph `START -> solve -> END` consuming canonical questions.

## Week 3 changes

| Area | Change |
|---|---|
| Paper metadata | `Paper` synced with upstream (`subject`, `class_name`, `board`); vision prompt extracts header metadata with null-tolerance; text path captures `Subject:`/`Class:`/board lines so PDF and image ingestion produce the same fields |
| Phone-photo preprocessing | Mild unsharp masking added to the existing normalize pipeline (orientation, downscale-if-huge, contrast); no binarization, no upscaling |
| Photo prompt rule | Skewed/rotated/shadowed/blurred photos must still be read carefully; truly unreadable content uses null + low confidence, never guesses |
| Budget overflow | Nested tiling (depth 2 max): a tile that itself overflows splits again; merges reuse the lossless overlap logic. Worst case 7 calls per page, only when truncating |
| Test fixtures | `scripts/make_bad_photos.py` builds deterministic bad-phone variants (rotation, blur, shadow, glare, shear, JPEG compression, downscaling, uneven light) from real paper pages, with ground truth (count + marks) taken from the digital text layer |
| Photo grading | `scripts/test_phone_photos.py` scores each photo on question count and per-question marks and prints a graded report |

## Image-ingestion flow (after Week 3)

```
photo file
 -> validate (type, size, integrity)
 -> preprocess (EXIF orientation, downscale, contrast, unsharp mask)
 -> Qwen vision extraction (JSON: metadata + questions)
 -> validate + normalize (never invent; null-tolerant)
 -> on token-budget overflow: split into overlapping tiles at whitespace
    gaps, extract, merge with option-union dedup (max depth 2)
 -> map to canonical Question (type mapping, options [] -> null, marks -> int)
 -> assemble canonical Paper (pap_<sha8> fingerprint, totals, sections,
    subject/class/board) -> downstream LangGraph
```

## Verification

### Offline suite

`pytest tests -m "not live"`: **52 passed** (schema conformance, validation,
merge/split logic, nested tiling, metadata capture, fixture determinism,
graph wiring, image validation).

### Bad-photo grading (live Groq vision, VM)

Set: 12 deterministic variants from real papers 455 (Maths) and 463
(Science). Ground truth from the digital text layer. Manifest honestly
labels these as synthetic degradations, a stand-in until real Set A phone
photos are available (no Set A fixtures exist in any team branch).

| Photo | Variant | Count | Marks | Notes |
|---|---|---|---|---|
| 455 p2 `jpeg_harsh` | heavy JPEG + contrast | 5/5 | 5/5 | PASS; header out of frame, metadata correctly null |
| 455 p2 `rot_blur` | rotated + blurred | 5/5 | 5/5 | PASS; recovered subject "Mathematics" from header edge |

Photos 3–12 were not yet graded (Groq free-tier daily quota exhausted during
testing; rerun `scripts/test_phone_photos.py --manifest <manifest>`).

### Downstream compatibility

Canonical questions carry exactly the keys the solve graph consumes
(`number/section/text/marks/type/options/has_figure/page/choice_group`,
plus paper `subject/class/board` as metadata). End-to-end
image → Paper → LangGraph → answers verified live in Week 2; Week 3
preserves those interfaces.

## Limitations

- Groq free tier: ~1000 output tokens per request (hence tiling) and
  200000 tokens/day shared across all testing; batch runs must be paced.
- One tile-boundary option was observed dropped on a dense page before the
  lossless-merge fix; the merge is now covered by regression tests but live
  re-confirmation on dense pages is still pending quota.
- Metadata is null when the header is out of frame — correct behavior, but
  full-paper metadata needs the first page.
- Superscript/combining-mark fidelity (e.g. `sₙ`, `0.45̄`) may simplify;
  proofread math before final use.
