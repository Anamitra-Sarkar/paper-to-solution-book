"""Score image ingestion on bad-phone photos against ground truth.

For each photo in a manifest (see make_bad_photos.py): run vision
extraction, compare question count and per-question marks, and print a
graded report. Needs GROQ_API_KEY.

Usage:
    GROQ_API_KEY=... python scripts/test_phone_photos.py --manifest test_runs/bad_phones/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade photo extraction vs ground truth.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sample", type=int, default=0,
                    help="only grade first N photos (0 = all)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set.")
        return 2

    from paper_to_solution.extractor import extract_questions_from_image

    manifest = json.loads(Path(args.manifest).read_text())
    base = Path(args.manifest).parent
    photos = manifest["photos"][:args.sample or None]
    rows = []
    for entry in photos:
        paper = extract_questions_from_image(base / entry["file"])
        got = {q.number: q.marks for q in paper.questions}
        exp = entry["expected_marks"]
        count_ok = len(got) == entry["expected_count"]
        marks_ok = sum(1 for n, m in exp.items() if got.get(n) == m)
        rows.append({"photo": entry["file"], "variant": entry["variant"],
                     "expected": entry["expected_count"], "got": len(got),
                     "count_ok": count_ok,
                     "marks_ok": f"{marks_ok}/{len(exp)}",
                     "all_ok": count_ok and marks_ok == len(exp),
                     "meta": [paper.subject, paper.class_name, paper.board]})
        r = rows[-1]
        print(f"{'PASS' if r['all_ok'] else 'FAIL'} {r['photo']}: "
              f"count {r['got']}/{r['expected']}, marks {r['marks_ok']}, "
              f"meta={r['meta']}")

    full = sum(1 for r in rows if r["all_ok"])
    print(f"\nFULL PASS: {full}/{len(rows)} "
          f"(count ok: {sum(1 for r in rows if r['count_ok'])}/{len(rows)})")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
    return 0 if full == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
