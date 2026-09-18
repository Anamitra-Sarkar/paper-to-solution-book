"""End-to-end helpers (Likhitha: integration & testing surface)."""
from __future__ import annotations

from pathlib import Path

from paper_to_solution.extractor import extract_questions_from_images
from paper_to_solution.graph import run_graph
from paper_to_solution.pdf_ingest import extract_questions_from_pdf


def run_image_to_answers(image_paths: list[str | Path], model: str | None = None) -> dict:
    """image(s) -> extract -> LangGraph solve -> answers. Full demo path."""
    extraction = extract_questions_from_images(image_paths, model=model)
    final = run_graph(extraction.questions) if extraction.questions else {
        "questions": [], "solutions": [], "errors": ["No questions extracted; nothing to solve."],
    }
    return {"extraction": extraction, "final_state": final}


def run_pdf_to_answers(pdf_path: str | Path, mode: str = "auto") -> dict:
    extraction = extract_questions_from_pdf(pdf_path, mode=mode)
    final = run_graph(extraction.questions) if extraction.questions else {
        "questions": [], "solutions": [], "errors": ["No questions extracted; nothing to solve."],
    }
    return {"extraction": extraction, "final_state": final}
