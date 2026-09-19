"""Random-paper verification: reproduce the acceptance test on any test paper.

Downloads a question paper from the shared test_papers set, checks
identification quality (count, numbering, marks, pages, options), runs a
sample of questions through the LangGraph answer path, and prints everything
a reviewer needs to grade correctness by eye.

Requires GROQ_API_KEY only for the answer sample; identification of digital
papers runs fully offline.

Examples:
    python scripts/test_random_paper.py --paper 464
    python scripts/test_random_paper.py --random --seed 7
    python scripts/test_random_paper.py --paper 464 --no-solve
    python scripts/test_random_paper.py --paper 464 --sample 6 --json-out report.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TEST_INDEX = "https://api.github.com/repos/yugal072/answer-book/contents/test_papers"
RAW_BASE = "https://raw.githubusercontent.com/yugal072/answer-book/main/test_papers"


def list_papers() -> list[str]:
    with urllib.request.urlopen(TEST_INDEX, timeout=30) as r:
        entries = json.load(r)
    return sorted(e["name"] for e in entries if e["name"].endswith(".pdf"))


def download(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{RAW_BASE}/{name}", dest)
    return dest


def pick_sample(questions, n: int):
    """Spread the sample across question types (MCQs first, then the rest)."""
    mcqs = [q for q in questions if q.type == "mcq"]
    rest = [q for q in questions if q.type != "mcq"]
    ordered = mcqs[:2] + rest + mcqs[2:]
    seen, out = set(), []
    for q in ordered:
        if q.number not in seen:
            seen.add(q.number)
            out.append(q)
        if len(out) == n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify extraction + answers on a test paper.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--paper", help="paper number, e.g. 464 for question_paper_464.pdf")
    g.add_argument("--random", action="store_true", help="pick a random paper")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "vision", "text"])
    ap.add_argument("--sample", type=int, default=4, help="how many questions to solve")
    ap.add_argument("--no-solve", action="store_true", help="identification check only")
    ap.add_argument("--json-out", default=None, help="write full extraction JSON here")
    ap.add_argument("--workdir", default="test_runs", help="download directory")
    args = ap.parse_args()

    papers = list_papers()
    print(f"Available papers: {len(papers)}")
    if args.random:
        rng = random.Random(args.seed)
        name = rng.choice(papers)
        print(f"Random pick (seed={args.seed}): {name}")
    else:
        name = f"question_paper_{args.paper}.pdf"
        if name not in papers:
            print(f"ERROR: {name} not in test set. Available: {papers}")
            return 2

    pdf = download(name, Path(args.workdir) / name)
    print(f"Downloaded: {pdf} ({pdf.stat().st_size // 1024} KB)")

    from paper_to_solution.pdf_ingest import extract_questions_from_pdf

    res = extract_questions_from_pdf(pdf, mode=args.mode)
    qs = res.questions
    print(f"\n### Identification ({args.mode}): {len(qs)} questions")

    failures = []
    if not qs:
        failures.append("no questions extracted at all")
    numbers = [q.number for q in qs]
    expected = [str(i) for i in range(1, len(qs) + 1)]
    if numbers != expected:
        failures.append(f"numbering not contiguous 1..{len(qs)}: {numbers}")
    missing_marks = [q.number for q in qs if q.marks is None]
    if missing_marks:
        failures.append(f"missing marks: {missing_marks}")
    print(f"Numbering: {'OK (1..%d contiguous)' % len(qs) if numbers == expected else 'CHECK ' + str(numbers)}")
    print(f"Marks: {'OK (all present)' if not missing_marks else 'MISSING ' + str(missing_marks)}")
    print(f"Pages: {sorted({q.page for q in qs})}")
    print(f"Types: {dict(Counter(q.type for q in qs))}")
    mcq_no_opts = [q.number for q in qs if q.type == "mcq" and not q.options]
    if mcq_no_opts:
        failures.append(f"MCQs without options: {mcq_no_opts}")
    print(f"MCQ options: {'OK' if not mcq_no_opts else 'MISSING for ' + str(mcq_no_opts)}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([q.model_dump() for q in qs], indent=1, ensure_ascii=False))
        print(f"Extraction JSON: {args.json_out}")

    if args.no_solve:
        solved, errors = [], []
    else:
        if not os.environ.get("GROQ_API_KEY"):
            print("\nGROQ_API_KEY not set: skipping answers (identification above is complete).")
            solved, errors = [], []
        else:
            from paper_to_solution.graph import run_graph

            sample = pick_sample(qs, args.sample)
            print(f"\n### Answers (sample {[q.number for q in sample]})")
            final = run_graph(sample)
            solved, errors = final["solutions"], final["errors"]
            by_no = {s["question_number"]: s for s in solved}
            for q in sample:
                print(f"\n----- Q{q.number} [{q.marks} marks, {q.type}] -----")
                print(q.text[:600])
                if q.options:
                    print("Options: " + " | ".join(q.options))
                s = by_no.get(q.number)
                print("ANSWER:" if s else "NO ANSWER:")
                print((s["answer"] if s else "")[:1500])
            if errors:
                print("\nErrors:", errors)
                failures.append(f"answer errors: {errors}")

    print("\n### Verdict")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        print("Grade answer correctness by eye from the sample above before submitting.")
        return 1
    print("PASS: identification clean" + (f", {len(solved)} answers generated" if solved else " (answers skipped)"))
    print("Grade answer correctness by eye from the sample above before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
