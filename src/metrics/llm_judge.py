from __future__ import annotations
import json
import re
from src.model import VLMWrapper

# The three quality dimensions we score each clarification question on
_DIMS = ("faithfulness", "reasonableness", "clarity")

# Prompt sent to the VLM judge along with the image.
# All candidates for one condition are scored in a single call so the judge
# has the same context for every candidate, making scores comparable.
_PROMPT = """\
You are evaluating clarification questions generated for an ambiguous visual question.
Look carefully at the image before scoring Faithfulness.

Ambiguous question: {ambiguous_question}

Candidates:
{candidates_text}

Score each candidate on these three dimensions (1-5):
  Faithfulness   — grounded in what is visible in the image, no unsupported objects or assumptions
  Reasonableness — addresses a plausible ambiguity in the original question
  Clarity        — specific and easy for a user to answer

Return ONLY a JSON object with key "candidates": a list of {n} objects in the same order, each with keys:
  "faithfulness", "reasonableness", "clarity"

Example for 2 candidates:
{{"candidates": [{{"faithfulness": 4, "reasonableness": 3, "clarity": 5}}, {{"faithfulness": 2, "reasonableness": 4, "clarity": 3}}]}}"""


def _extract_json_field(text: str, field: str):
    # Find the first JSON object in the raw model output and return the requested field
    try:
        start = text.find("{")
        if start != -1:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if field in obj:
                return obj[field]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_score(val) -> int | None:
    # Convert a raw value to an int in [1, 5], or None if invalid
    try:
        v = int(val)
        return v if 1 <= v <= 5 else None
    except (TypeError, ValueError):
        return None


def llm_judge_candidates(
    candidates: list[str],
    ambiguous_question: str,
    image_path: str,
    model: VLMWrapper,
) -> dict:
    """Score all candidates from one condition on 3 dimensions in a single LLM call.

    Returns a dict with:
      - "candidates": per-candidate scores for each dimension + their mean
      - "mean_faithfulness/reasonableness/clarity": average score across all candidates
      - "raw_output": raw model response for debugging
    """
    if not candidates:
        return {"candidates": [], "mean_faithfulness": None,
                "mean_reasonableness": None, "mean_clarity": None, "raw_output": ""}

    # Format candidates as a numbered list for the prompt
    candidates_text = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    prompt = _PROMPT.format(
        n=len(candidates),
        ambiguous_question=ambiguous_question,
        candidates_text=candidates_text,
    )

    # Single LLM call — image is passed so the judge can verify Faithfulness visually
    raw = model.generate(image_path, prompt, max_new_tokens=512)

    # Parse the "candidates" list from the JSON response
    raw_list = _extract_json_field(raw, "candidates")
    if not isinstance(raw_list, list):
        raw_list = []
    if len(raw_list) != len(candidates):
        print(f"Warning: expected {len(candidates)} scored candidates, got {len(raw_list)}")

    # Build per-candidate result entries; fall back to {} if the model missed a candidate
    results = []
    for idx, cand in enumerate(candidates):
        obj = raw_list[idx] if idx < len(raw_list) else {}
        entry: dict = {"candidate": cand}
        for dim in _DIMS:
            entry[dim] = _parse_score(obj.get(dim))
        valid = [entry[d] for d in _DIMS if entry[d] is not None]
        entry["mean"] = round(sum(valid) / len(valid), 3) if valid else None
        results.append(entry)

    # Compute per-dimension averages across all candidates
    dim_means = {
        f"mean_{dim}": round(sum(v for r in results if (v := r[dim]) is not None) /
                             max(1, sum(1 for r in results if r[dim] is not None)), 3)
        if any(r[dim] is not None for r in results) else None
        for dim in _DIMS
    }

    return {"candidates": results, **dim_means, "raw_output": raw}