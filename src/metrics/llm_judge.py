from __future__ import annotations
import json
from src.model import VLMWrapper

_DIMS = ("faithfulness", "reasonableness")

_PROMPTS = {
    "faithfulness": """\
You are evaluating whether a clarification question is grounded in what is visible in the image.
You are shown an image and an ambiguous question a user asked about it.

Ambiguous question: {ambiguous_question}

Candidates:
{candidates_text}

For each candidate score Faithfulness (1-5):
Does THIS candidate only refer to objects, people, or details actually visible in the image?
Does it avoid introducing assumptions or details not shown?
(1 = refers to things not visible in the image, 5 = perfectly grounded in what is visible)

Be critical. Reserve 5 for questions that are clearly and specifically grounded in the image.

Return ONLY a JSON object with no extra text:
{{"candidates": [{{"f_score": <1-5>, "note": "<one sentence about THIS candidate's visual grounding>"}}]}}
The list must contain exactly {n} objects in the same order as the candidates above.""",

    "reasonableness": """\
You are evaluating whether a clarification question helps resolve an ambiguous visual question.
You are shown an image and an ambiguous question a user asked about it.

Ambiguous question: {ambiguous_question}

Candidates:
{candidates_text}

For each candidate score Reasonableness (1-5):
Does THIS candidate ask about the right thing to resolve what the user could have meant?
A good question narrows down the user's intent by distinguishing between plausible interpretations of the ambiguous question.
(1 = asks about something unrelated to the ambiguity, 5 = directly identifies and targets the core ambiguity)

Be critical. Reserve 5 for questions that clearly and specifically target the ambiguity.

Return ONLY a JSON object with no extra text:
{{"candidates": [{{"r_score": <1-5>, "note": "<one sentence about THIS candidate's relevance to the ambiguity>"}}]}}
The list must contain exactly {n} objects in the same order as the candidates above.""",
}


def _extract_json_field(text: str, field: str):
    try:
        start = text.find("{")
        if start != -1:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict) and field in obj:
                return obj[field]
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        start = text.find("[")
        if start != -1:
            arr, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(arr, list):
                return arr
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
    """Score all candidates on faithfulness and reasonableness using one LLM call per dimension.

    Returns a dict with:
      - "candidates": per-candidate scores + notes for each dimension + their mean
      - "mean_faithfulness", "mean_reasonableness": averages across all candidates
      - "raw_outputs": raw model responses per dimension for debugging
    """
    if not candidates:
        return {"candidates": [], "mean_faithfulness": None,
                "mean_reasonableness": None, "raw_outputs": {}}

    candidates_text = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))

    # One LLM call per dimension so the judge focuses on one criterion at a time
    raw_outputs: dict[str, str] = {}
    dim_raw_lists: dict[str, list] = {}
    for dim, prompt_template in _PROMPTS.items():
        prompt = prompt_template.format(
            n=len(candidates),
            ambiguous_question=ambiguous_question,
            candidates_text=candidates_text,
        )
        raw = model.generate(image_path, prompt, max_new_tokens=1024)
        raw_outputs[dim] = raw
        raw_list = _extract_json_field(raw, "candidates")
        if not isinstance(raw_list, list):
            raw_list = []
        if len(raw_list) != len(candidates):
            print(f"Warning [{dim}]: expected {len(candidates)} scored candidates, got {len(raw_list)}")
        dim_raw_lists[dim] = raw_list

    # Merge per-candidate entries across dimensions
    score_keys = {"faithfulness": "f_score", "reasonableness": "r_score"}
    results = []
    for idx, cand in enumerate(candidates):
        entry: dict = {"candidate": cand}
        for dim in _DIMS:
            obj = dim_raw_lists[dim][idx] if idx < len(dim_raw_lists[dim]) else {}
            entry[dim] = _parse_score(obj.get(score_keys[dim]))
            entry[f"{dim}_note"] = obj.get("note") or ""
        valid = [entry[d] for d in _DIMS if entry[d] is not None]
        entry["mean"] = round(sum(valid) / len(valid), 3) if valid else None
        results.append(entry)

    dim_means = {
        f"mean_{dim}": round(sum(v for r in results if (v := r[dim]) is not None) /
                             max(1, sum(1 for r in results if r[dim] is not None)), 3)
        if any(r[dim] is not None for r in results) else None
        for dim in _DIMS
    }

    return {"candidates": results, **dim_means, "raw_outputs": raw_outputs}