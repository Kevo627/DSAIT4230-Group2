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


# Judge 1 — Candidate selection from N samples (image-grounded, no ground truth)

_CANDIDATE_SCORING_PROMPT = """\
You are evaluating clarification questions for a visually ambiguous query.

The user asked an ambiguous question about an image. Below are {n} candidate \
clarification questions generated to resolve that ambiguity.

User question: {ambiguous_question}

Score each candidate on a scale from 1 to 5 based on how well it would resolve \
the referential ambiguity visible in the image, considering:
  - Faithfulness: is the question grounded in what is actually in the image?
  - Reasonableness: does it target a real ambiguity in the question?
  - Clarity: is it specific and easy for a user to answer?

Candidates:
{candidates_text}

Return ONLY a JSON object with integer scores 1-5 for each candidate index (0-based).
Example for 3 candidates: {{"scores": [4, 2, 5]}}"""


def llm_judge_candidates(
    candidates: list[str],
    reference_question: str,
    image_path: str,
    ambiguous_question: str,
    model: VLMWrapper,
) -> dict:
    if not candidates:
        return {
            "best_candidate": "",
            "best_index": None,
            "best_score": 0.0,
            "scores": [],
        }

    candidates_text = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    prompt = _CANDIDATE_SCORING_PROMPT.format(
        n=len(candidates),
        ambiguous_question=ambiguous_question,
        candidates_text=candidates_text,
    )

    raw = model.generate(image_path, prompt, max_new_tokens=256)

    raw_scores: list[int | None] = []
    scores_field = _extract_json_field(raw, "scores")
    if isinstance(scores_field, list):
        for s in scores_field:
            try:
                val = int(s)
                raw_scores.append(val if 1 <= val <= 5 else None)
            except (TypeError, ValueError):
                raw_scores.append(None)

    while len(raw_scores) < len(candidates):
        raw_scores.append(None)
    raw_scores = raw_scores[: len(candidates)]

    float_scores = [float(s) if s is not None else 1.0 for s in raw_scores]
    scored = [{"candidate": c, "score": s} for c, s in zip(candidates, float_scores)]
    best_index = int(max(range(len(float_scores)), key=lambda i: float_scores[i]))

    return {
        "best_candidate": candidates[best_index],
        "best_index": best_index,
        "best_score": float_scores[best_index],
        "scores": scored,
    }


# Judge 2 — Post-hoc 1 sample evaluation 

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