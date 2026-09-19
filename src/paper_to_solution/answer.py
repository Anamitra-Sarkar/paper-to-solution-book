"""Answer generation: canonical Question -> answer.

Kept strictly separate from extraction: this module never sees images,
only canonical question dicts/objects (keys: number/text/type/marks/options,
the same representation PDF ingestion produces). Default model is a Groq
text LLM.
"""
from __future__ import annotations

from groq import Groq

from paper_to_solution import config
from paper_to_solution.canonical import Question
from paper_to_solution.schemas import SolvedQuestion

ANSWER_SYSTEM_PROMPT = """You are solving exam questions from a question paper. \
Answer each question directly, accurately and completely. \
Show working for numerical problems. Keep formatting simple plain text."""


def _client() -> Groq:
    return Groq(api_key=config.get_groq_api_key(), timeout=config.GROQ_TIMEOUT_S)


def answer_question(question: Question, model: str | None = None) -> SolvedQuestion:
    model = model or config.ANSWER_MODEL
    header = (
        f"Question {question.number} ({question.marks} marks):"
        if question.marks else f"Question {question.number}:"
    )
    parts = [header, question.text]
    if question.options:
        parts.append("Options:")
        parts.extend(f"- {o}" for o in question.options)
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ],
    )
    return SolvedQuestion(
        question_number=question.number,
        question_text=question.text,
        answer=(resp.choices[0].message.content or "").strip(),
        model=model,
    )


def answer_questions(questions: list[Question], model: str | None = None) -> list[SolvedQuestion]:
    return [answer_question(q, model=model) for q in questions]
