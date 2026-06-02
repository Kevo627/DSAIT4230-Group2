

import math
from src.model import VLMWrapper, parse_json_output



_INTENT_PROMPT = """You are analyzing a visual question that is ambiguous due to intent underspecification.

The user asked: "{ambiguous_question}"

List {n} distinct, plausible user goals behind this question. Each goal should describe a different specific aspect the user might care about. Write each as a full sentence starting with "The user wants to know".

Return ONLY a JSON object, no extra text:
{{
    "intents": [
        "The user wants to know ...",
        "The user wants to know ...",
        ...
    ]
}}"""


def generate_intents(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    n: int = 4,
) -> list[str]:
    """Generate n plausible user intents for an ambiguous question."""
    prompt = _INTENT_PROMPT.format(ambiguous_question=ambiguous_question, n=n)
    raw = model.generate(image_path, prompt)
    parsed = parse_json_output(raw)
    intents = parsed.get("intents", [])
    if not isinstance(intents, list) or len(intents) == 0:
        return [raw.strip()]
    return [str(i) for i in intents[:n]]



_DISAMBIGUATION_PROMPT = """You are evaluating how well a clarification question disambiguates user intent.

Original ambiguous question: "{ambiguous_question}"
Clarification question being evaluated: "{cq}"

Possible user intents:
{intents_block}

How many of these {n} intents would give DISTINCTLY DIFFERENT answers to the \
clarification question? Count only intents whose responses would clearly differ \
from each other — not just different wording, but different meaning.

Return ONLY a JSON object, no extra text:
{{
    "distinct_responses": <integer from 1 to {n}>,
    "reasoning": "..."
}}"""


def score_cq_disambiguation(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq: str,
    intents: list[str],
) -> dict:
    """
    Score a CQ by asking the VLM how many intents would give distinctly
    different answers to it. Higher = more informative CQ.
    Returns a dict with the score and reasoning for traceability.
    """
    intents_block = "\n".join(f"  {i + 1}. {intent}" for i, intent in enumerate(intents))
    prompt = _DISAMBIGUATION_PROMPT.format(
        ambiguous_question=ambiguous_question,
        cq=cq,
        intents_block=intents_block,
        n=len(intents),
    )
    raw = model.generate(image_path, prompt)
    parsed = parse_json_output(raw)

    distinct = parsed.get("distinct_responses", 1)
    if not isinstance(distinct, int):
        try:
            distinct = int(distinct)
        except (ValueError, TypeError):
            distinct = 1
    distinct = max(1, min(distinct, len(intents)))

    return {
        "cq": cq,
        "disambiguation_score": distinct,
        "reasoning": parsed.get("reasoning", ""),
    }



def select_best_cq(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq_candidates: list[str],
    n_intents: int = 4,
) -> dict:
    """
    Build the intent space, score each candidate CQ by disambiguation score,
    return the best. Ties broken by first occurrence.
    """
    intents = generate_intents(model, image_path, ambiguous_question, n=n_intents)

    scores = [
        score_cq_disambiguation(model, image_path, ambiguous_question, cq, intents)
        for cq in cq_candidates
    ]

    best = max(scores, key=lambda s: s["disambiguation_score"])

    return {
        "intents": intents,
        "candidate_scores": scores,
        "best_cq": best["cq"],
        "best_disambiguation_score": best["disambiguation_score"],
    }



_USER_RESPONSE_PROMPT = """You are a user who asked: "{ambiguous_question}"

You were asked this clarification question: "{cq}"

The clarification question asks you to confirm what you want to know. Answer it directly — typically yes or no, followed by a brief clarification of your actual goal if needed.

Example: if asked "Are you asking about the architectural style of the building, or its historical background?", a good response is: "I am asking about its historical background."

Return ONLY a JSON object, no extra text:
{{
    "response": "..."
}}"""


def simulate_user_response(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq: str,
) -> str:
    """
    Simulate a user's response to the selected CQ.
    No intent or gold data used — the VLM responds based on the image alone.
    """
    prompt = _USER_RESPONSE_PROMPT.format(
        ambiguous_question=ambiguous_question,
        cq=cq,
    )
    raw = model.generate(image_path, prompt)
    parsed = parse_json_output(raw)
    return parsed.get("response", raw.strip())



_FINAL_ANSWER_PROMPT = """\
You are answering a visual question about an image.

The user's original question: "{ambiguous_question}"
A clarification question was asked: "{cq}"
The user's response to the clarification: "{user_response}"

Based on the clarification, answer the original question as specifically \
and concisely as possible.

Return ONLY a JSON object, no extra text:
{{
    "answer": "..."
}}"""


def generate_final_answer(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq: str,
    user_response: str,
) -> str:
    """Generate the final answer conditioned on (image, q, cq, user_response)."""
    prompt = _FINAL_ANSWER_PROMPT.format(
        ambiguous_question=ambiguous_question,
        cq=cq,
        user_response=user_response,
    )
    raw = model.generate(image_path, prompt)
    parsed = parse_json_output(raw)
    return parsed.get("answer", raw.strip())