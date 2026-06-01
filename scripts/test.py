"""
Smoke test: run one example through the full pipeline and print every step.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --condition at_cot --n_candidates 3
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_intent_examples
from src.model import VLMWrapper
from src.conditions.standard import StandardCondition
from src.conditions.at import ATCondition
from src.conditions.cot import CoTCondition
from src.conditions.at_cot import ATCoTCondition
from src.uot import (
    generate_intents,
    select_best_cq,
    simulate_user_response,
    generate_final_answer,
)
from scripts.run_pipeline import generate_candidates

ALL_CONDITIONS = {
    "standard": StandardCondition,
    "at": ATCondition,
    "cot": CoTCondition,
    "at_cot": ATCoTCondition,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(ALL_CONDITIONS.keys()), default="standard")
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--n_intents", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    examples = load_intent_examples(limit=1)
    example = examples[0]

    print("=" * 70)
    print(f"ID                  : {example['id']}")
    print(f"Ambiguous question  : {example['ambiguous_question']}")
    print(f"Gold clarification  : {example['gold_clarification']}")
    print(f"Gold answer         : {example['gold_answer']}")
    print(f"Image path          : {example['image_path']}")
    print(f"Image exists        : {os.path.exists(example['image_path'])}")

    model = VLMWrapper(model_name=args.model) if args.model else VLMWrapper()
    condition = ALL_CONDITIONS[args.condition](model)

    print(f"\n[1] Generating {args.n_candidates} CQ candidates ({args.condition}) ...")
    candidates = generate_candidates(
        model, condition, example,
        args.n_candidates, args.temperature, args.top_p,
    )
    for i, c in enumerate(candidates):
        print(f"  Candidate {i}: {c}")

    print(f"\n[2] Generating {args.n_intents} intents ...")
    intents = generate_intents(model, example["image_path"], example["ambiguous_question"], n=args.n_intents)
    for i, intent in enumerate(intents):
        print(f"  Intent {i}: {intent}")

    print("\n[3] Scoring candidates by expected information gain ...")
    uot_result = select_best_cq(
        model, example["image_path"], example["ambiguous_question"],
        candidates, n_intents=args.n_intents,
    )
    for s in uot_result["candidate_scores"]:
        print(f"  score={s['disambiguation_score']}  CQ: {s['cq']}")
        print(f"    reasoning: {s['reasoning']}")

    print(f"\n[4] Selected CQ (score={uot_result['best_disambiguation_score']}): {uot_result['best_cq']}")

    print("\n[5] Simulating user response ...")
    user_response = simulate_user_response(
        model, example["image_path"], example["ambiguous_question"], uot_result["best_cq"]
    )
    print(f"  User response: {user_response}")

    print("\n[6] Generating final answer ...")
    final_answer = generate_final_answer(
        model, example["image_path"], example["ambiguous_question"],
        uot_result["best_cq"], user_response,
    )
    print(f"  Final answer : {final_answer}")
    print(f"  Gold answer  : {example['gold_answer']}")
    print("=" * 70)


if __name__ == "__main__":
    main()