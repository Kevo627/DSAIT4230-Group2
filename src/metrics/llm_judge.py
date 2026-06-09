from __future__ import annotations
import json
import re
from src.model import VLMWrapper


def _extract_score(text: str, lo: int = 1, hi: int = 5) -> int | None:
    for tok in re.findall(r"\d+", text):
        val = int(tok)
        if lo <= val <= hi:
            return val
    return None


def _extract_json_field(text: str, field: str):
    try:
        start = text.find("{")
        if start != -1:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if field in obj:
                return obj[field]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# Post-hoc evaluation — score one clarification question on three dimensions

_QUALITY_PROMPT = """\
You are evaluating a clarification question asked after seeing an image and an \
ambiguous user question.

Ambiguous question: {ambiguous_question}
Clarification question: {clarification_question}

Score on a scale of 1 to 5 for each dimension:
  Faithfulness: is the CQ grounded in the image without introducing unsupported objects or assumptions?
  Reasonableness: does it address a plausible ambiguity in the original question?
  Clarity: is it specific and easy for a user to answer?

Return ONLY a JSON object: {{"faithfulness": <1-5>, "reasonableness": <1-5>, "clarity": <1-5>}}"""


def llm_judge_quality(
    clarification_question: str,
    ambiguous_question: str,
    image_path: str,
    model: VLMWrapper,
) -> dict:
    prompt = _QUALITY_PROMPT.format(
        ambiguous_question=ambiguous_question,
        clarification_question=clarification_question,
    )
    raw = model.generate(image_path, prompt, max_new_tokens=64)

    scores: dict[str, int | None] = {}
    for dim in ("faithfulness", "reasonableness", "clarity"):
        val = _extract_json_field(raw, dim)
        if val is not None:
            try:
                int_val = int(val)
                scores[dim] = int_val if 1 <= int_val <= 5 else None
            except (TypeError, ValueError):
                scores[dim] = None
        else:
            scores[dim] = _extract_score(raw)

    valid = [v for v in scores.values() if v is not None]
    mean = sum(valid) / len(valid) if valid else None

    return {
        "faithfulness": scores.get("faithfulness"),
        "reasonableness": scores.get("reasonableness"),
        "clarity": scores.get("clarity"),
        "mean": mean,
        "raw_output": raw,
    }
