"""Answer generation: question -> answer (Abhiram).

Kept strictly separate from extraction: this module never sees images,
only ExtractedQuestion text. Default model is a Groq text LLM.
"""
from __future__ import annotations

from groq import Groq

from paper_to_solution import config
from paper_to_solution.schemas import ExtractedQuestion, SolvedQuestion

ANSWER_SYSTEM_PROMPT = """You are solving exam questions from a question paper. \
Answer each question directly, accurately and completely. \
Show working for numerical problems. Keep formatting simple plain text."""


def _client() -> Groq:
    return Groq(api_key=config.get_groq_api_key(), timeout=config.GROQ_TIMEOUT_S)


def answer_question(question: ExtractedQuestion, model: str | None = None) -> SolvedQuestion:
    model = model or config.ANSWER_MODEL
    parts = [f"Question {question.question_number} ({question.marks} marks):" if question.marks else f"Question {question.question_number}:"]
    parts.append(question.question_text)
    if question.options:
        parts.append("Options:")
        parts.extend(f"- {o}" for o in question.options)
    for s in question.subquestions:
        parts.append(f"({s.label}) {s.text}")
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
        question_number=question.question_number,
        question_text=question.question_text,
        answer=(resp.choices[0].message.content or "").strip(),
        model=model,
    )


def answer_questions(questions: list[ExtractedQuestion], model: str | None = None) -> list[SolvedQuestion]:
    return [answer_question(q, model=model) for q in questions]
