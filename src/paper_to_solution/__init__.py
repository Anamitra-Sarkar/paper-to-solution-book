"""Paper-to-Solution-Book: image/PDF question extraction -> LangGraph -> answers.

Both ingestion paths return the canonical Paper/Question schema (canonical.py,
identical to the team-wide models/loaders_models.py contract).
"""
from paper_to_solution.canonical import Paper, Question, to_canonical_paper, to_canonical_question
from paper_to_solution.schemas import ExtractedQuestion, ExtractionResult, SolvedQuestion, PipelineState
from paper_to_solution.extractor import extract_questions_from_image, extract_questions_from_images
from paper_to_solution.pipeline import run_image_to_answers, run_pdf_to_answers

__all__ = [
    "Paper",
    "Question",
    "to_canonical_paper",
    "to_canonical_question",
    "ExtractedQuestion",
    "ExtractionResult",
    "SolvedQuestion",
    "PipelineState",
    "extract_questions_from_image",
    "extract_questions_from_images",
    "run_image_to_answers",
    "run_pdf_to_answers",
]
