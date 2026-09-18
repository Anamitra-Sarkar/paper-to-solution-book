"""LangGraph pipeline (Yashwanth): START -> solve -> END.

Consumes ExtractedQuestion list (from image OCR or PDF ingestion),
routes each through answer generation, records per-question errors
without aborting the whole run.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from paper_to_solution.answer import answer_question
from paper_to_solution.schemas import ExtractedQuestion, SolvedQuestion


class GraphState(TypedDict):
    questions: list[dict]
    solutions: list[dict]
    errors: list[str]


def _solve_node(state: GraphState) -> GraphState:
    solutions: list[dict] = list(state.get("solutions", []))
    errors: list[str] = list(state.get("errors", []))
    for qd in state.get("questions", []):
        try:
            q = ExtractedQuestion(**qd)
            solved: SolvedQuestion = answer_question(q)
            solutions.append(solved.model_dump())
        except Exception as e:
            errors.append(f"Q{qd.get('question_number', '?')}: {type(e).__name__}: {e}")
    return {"questions": state.get("questions", []), "solutions": solutions, "errors": errors}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("solve", _solve_node)
    g.add_edge(START, "solve")
    g.add_edge("solve", END)
    return g.compile()


def run_graph(questions: list[ExtractedQuestion]) -> GraphState:
    """Pass extracted questions through START -> solve -> END."""
    app = build_graph()
    return app.invoke(
        {"questions": [q.model_dump() for q in questions], "solutions": [], "errors": []}
    )
