"""Shared structured schemas (single source of truth for all team members).

Anamitra (image OCR) produces ExtractionResult.
Yashwanth (LangGraph) threads PipelineState through START -> solve -> END.
Abhiram (answers) consumes ExtractedQuestion, produces SolvedQuestion.
Yugal (PDF) renders pages -> reuses the image extractor page by page.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QuestionType = Literal[
    "mcq",
    "short_answer",
    "descriptive",
    "numerical",
    "coding",
    "fill_in_the_blank",
    "true_false",
    "unknown",
]


class SubQuestion(BaseModel):
    label: str = Field(description="e.g. 'a', 'b', '(i)', '1(a)'")
    text: str = ""
    marks: int | float | None = None


class ExtractedQuestion(BaseModel):
    question_number: str = Field(description="As printed, e.g. '1', '2(a)', 'Q3'. Never invented.")
    question_text: str = ""
    marks: int | float | None = None
    question_type: QuestionType = "unknown"
    options: list[str] = Field(default_factory=list)
    subquestions: list[SubQuestion] = Field(default_factory=list)
    source_page: int = 1
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_notes: str | None = None
    # Canonical-bound fields (see canonical.py): populated when visible,
    # otherwise None/False per upstream conventions. Never fabricated.
    section: str | None = None
    has_figure: bool = False


class ExtractionResult(BaseModel):
    questions: list[ExtractedQuestion] = Field(default_factory=list)
    raw_response: str | None = Field(default=None, description="Raw model text, kept for debugging")
    model: str = ""


class SolvedQuestion(BaseModel):
    question_number: str = ""
    question_text: str = ""
    answer: str = ""
    model: str = ""


class PipelineState(BaseModel):
    """LangGraph state: START -> solve -> END."""

    questions: list[ExtractedQuestion] = Field(default_factory=list)
    solutions: list[SolvedQuestion] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def to_langgraph_dict(self) -> dict[str, Any]:
        return {
            "questions": [q.model_dump() for q in self.questions],
            "solutions": [s.model_dump() for s in self.solutions],
            "errors": list(self.errors),
        }

    @classmethod
    def from_langgraph_dict(cls, d: dict[str, Any]) -> "PipelineState":
        return cls(
            questions=[ExtractedQuestion(**q) for q in d.get("questions", [])],
            solutions=[SolvedQuestion(**s) for s in d.get("solutions", [])],
            errors=list(d.get("errors", [])),
        )
