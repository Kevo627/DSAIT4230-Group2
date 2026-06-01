"""
Uncertainty-of-Thoughts (UoT) utilities for intent-underspecified VQA.

Pipeline stages implemented here:
  1. generate_intents        — build intent possibility space from (image, question)
  2. score_cq                — compute expected information gain for one CQ candidate
     ├── simulate_response   — VLM-simulate user reply per intent
     ├── group_responses     — cluster responses into distinguishable groups
     └── compute_ig          — entropy reduction from grouping
  3. select_best_cq          — score all candidates, return best
  4. simulate_user_response  — oracle-free user response to selected CQ
  5. generate_final_answer   — final answer conditioned on (image, q, cq, response)
"""

import math
from src.model import VLMWrapper, parse_json_output



_INTENT_PROMPT = """\
You are analyzing a visual question that is ambiguous due to intent underspecification.

The user asked: "{ambiguous_question}"

List {n} distinct, plausible interpretations of what the user wants to know about \
the image. Each interpretation should represent a different user goal or aspect.

Return ONLY a JSON object, no extra text:
{{
    "intents": ["...", "...", ...]
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



_SIMULATE_RESPONSE_PROMPT = """\
You are a user looking at this image. You asked: "{ambiguous_question}"

You were then asked the following clarification question: "{cq}"

Your actual underlying goal is: "{intent}"

Answer the clarification question briefly and naturally, as a real user would. \
Do not reveal your underlying goal explicitly.

Return ONLY a JSON object, no extra text:
{{
    "response": "..."
}}"""


def simulate_response(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq: str,
    intent: str,
) -> str:
    """Simulate what a user with a specific intent would say in response to a CQ."""
    prompt = _SIMULATE_RESPONSE_PROMPT.format(
        ambiguous_question=ambiguous_question,
        cq=cq,
        intent=intent,
    )
    raw = model.generate(image_path, prompt)
    parsed = parse_json_output(raw)
    return parsed.get("response", raw.strip())



_GROUP_PROMPT = """\
A user asked an ambiguous question and was given a clarification question. \
Below are responses from users with different underlying goals.

Clarification question: "{cq}"

Responses (indexed from 0):
{responses_block}

Group these responses by whether they convey the same type of answer — \
i.e., a system reading only the response could not distinguish between \
the users in the same group.

Return ONLY a JSON object where keys are group IDs (integers starting from 0) \
and values are lists of response indices. Every index (0 to {last_idx}) must \
appear in exactly one group. No extra text:
{{
    "groups": {{"0": [<indices>], "1": [<indices>], ...}}
}}"""


def group_responses(
    model: VLMWrapper,
    image_path: str,
    cq: str,
    responses: list[str],
) -> list[int]:
    """
    Cluster simulated responses into distinguishable groups using the VLM.

    Returns a list of group IDs, one per response (same length as `responses`).
    E.g. [0, 1, 0, 2] means responses 0 and 2 are in the same group.
    Falls back to all-distinct groups on parse failure.
    """
    responses_block = "\n".join(f'  {i}: "{r}"' for i, r in enumerate(responses))
    prompt = _GROUP_PROMPT.format(
        cq=cq,
        responses_block=responses_block,
        last_idx=len(responses) - 1,
    )
    raw = model.generate(image_path, prompt)
    parsed = parse_json_output(raw)
    groups_dict = parsed.get("groups", {})

    group_ids = list(range(len(responses)))  # fallback: all distinct
    if isinstance(groups_dict, dict) and len(groups_dict) > 0:
        assigned = [False] * len(responses)
        candidate = list(range(len(responses)))
        for gid_str, indices in groups_dict.items():
            if not isinstance(indices, list):
                continue
            try:
                gid = int(gid_str)
            except ValueError:
                continue
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(responses):
                    candidate[idx] = gid
                    assigned[idx] = True
        # only accept if every index was assigned
        if all(assigned):
            group_ids = candidate

    return group_ids



def compute_ig(group_ids: list[int], n_intents: int) -> float:
    """
    Expected information gain of a CQ given the response partition.

    With a uniform prior P(i_k) = 1/N:
      H_prior = log(N)
      E[H_posterior | response] = sum_g (|g|/N) * log(|g|)
      IG = H_prior - E[H_posterior | response]

    A CQ where every intent lands in its own group → IG = log(N) (maximum).
    A CQ where all intents land in one group     → IG = 0 (uninformative).
    """
    if n_intents <= 1:
        return 0.0

    h_prior = math.log(n_intents)

    group_sizes: dict[int, int] = {}
    for gid in group_ids:
        group_sizes[gid] = group_sizes.get(gid, 0) + 1

    expected_posterior_h = sum(
        (size / n_intents) * math.log(size)
        for size in group_sizes.values()
        if size > 0
    )

    return h_prior - expected_posterior_h



def score_cq(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq: str,
    intents: list[str],
) -> dict:
    """
    Score a single CQ candidate by expected IG over the intent space.

    Returns a dict with the score and all intermediate outputs for traceability.
    """
    responses = [
        simulate_response(model, image_path, ambiguous_question, cq, intent)
        for intent in intents
    ]
    group_ids = group_responses(model, image_path, cq, responses)
    ig = compute_ig(group_ids, len(intents))
    return {
        "cq": cq,
        "ig": ig,
        "simulated_responses": responses,
        "group_ids": group_ids,
    }


def select_best_cq(
    model: VLMWrapper,
    image_path: str,
    ambiguous_question: str,
    cq_candidates: list[str],
    n_intents: int = 4,
) -> dict:
    """
    Build the intent space, score each candidate CQ by IG, return the best.

    Returns a dict with intents, per-candidate scores, and the selected CQ.
    """
    intents = generate_intents(model, image_path, ambiguous_question, n=n_intents)

    scores = [
        score_cq(model, image_path, ambiguous_question, cq, intents)
        for cq in cq_candidates
    ]

    best = max(scores, key=lambda s: s["ig"])

    return {
        "intents": intents,
        "candidate_scores": scores,
        "best_cq": best["cq"],
        "best_ig": best["ig"],
    }



_USER_RESPONSE_PROMPT = """\
You are a user looking at this image. You asked: "{ambiguous_question}"

You were then asked the following clarification question: "{cq}"

Answer the clarification question briefly and naturally, based only on \
what you see in the image.

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