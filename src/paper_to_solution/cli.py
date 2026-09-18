"""CLI: ptsb extract <image...> | ptsb solve <image...> | ptsb serve"""
from __future__ import annotations

import argparse
import json
import os


def _as_json(obj) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), indent=2, ensure_ascii=False)
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(prog="ptsb", description="Paper to Solution Book prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="image(s)/PDF -> extracted questions JSON")
    p_ex.add_argument("inputs", nargs="+", help="image files (.png/.jpg/.webp) or a .pdf")

    p_so = sub.add_parser("solve", help="image(s) -> extract -> LangGraph -> answers")
    p_so.add_argument("inputs", nargs="+")

    p_se = sub.add_parser("serve", help="run demo web UI")
    p_se.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))

    args = ap.parse_args()
    if args.cmd == "extract":
        from paper_to_solution.extractor import extract_questions_from_images
        from paper_to_solution.pdf_ingest import extract_questions_from_pdf

        if len(args.inputs) == 1 and args.inputs[0].lower().endswith(".pdf"):
            print(_as_json(extract_questions_from_pdf(args.inputs[0])))
        else:
            print(_as_json(extract_questions_from_images(args.inputs)))
    elif args.cmd == "solve":
        from paper_to_solution.pipeline import run_image_to_answers, run_pdf_to_answers

        if len(args.inputs) == 1 and args.inputs[0].lower().endswith(".pdf"):
            out = run_pdf_to_answers(args.inputs[0])
        else:
            out = run_image_to_answers(args.inputs)
        print(json.dumps({
            "questions": [q.model_dump() for q in out["extraction"].questions],
            "solutions": out["final_state"]["solutions"],
            "errors": out["final_state"]["errors"],
        }, indent=2, ensure_ascii=False))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("paper_to_solution.app:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
