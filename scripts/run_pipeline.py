

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_intent_examples
from src.model import VLMWrapper, parse_json_output, DEFAULT_MODEL
from src.conditions.standard import StandardCondition
from src.conditions.at import ATCondition
from src.conditions.cot import CoTCondition
from src.conditions.at_cot import ATCoTCondition
from src.uot import select_best_cq, simulate_user_response, generate_final_answer

ALL_CONDITIONS = {
    "standard": StandardCondition,
    "at": ATCondition,
    "cot": CoTCondition,
    "at_cot": ATCoTCondition,
}

DEFAULT_OUTPUT = os.path.join("results", "pipeline_results.jsonl")


def generate_candidates(
    model: VLMWrapper,
    condition,
    example: dict,
    n: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """
    Sample n CQ candidates from a condition with do_sample=True.
    Deduplicates while preserving generation order.
    """
    prompt = condition.build_prompt(example["ambiguous_question"])
    seen: set[str] = set()
    candidates: list[str] = []

    def _normalise(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]", "", s.lower())

    for _ in range(n):
        raw = model.generate(
            example["image_path"],
            prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        parsed = parse_json_output(raw)
        cq = parsed.get("clarification_question", "").strip()
        key = _normalise(cq)
        if cq and key not in seen:
            seen.add(key)
            candidates.append(cq)

    return candidates


def run_example(
    model: VLMWrapper,
    condition,
    example: dict,
    n_candidates: int,
    n_intents: int,
    temperature: float,
    top_p: float,
) -> dict:
    # 1. Generate CQ candidates
    candidates = generate_candidates(
        model, condition, example, n_candidates, temperature, top_p
    )
    if not candidates:
        raise ValueError("No valid CQ candidates generated")

    # 2-4. UoT: build intent space, score candidates, select best CQ
    uot_result = select_best_cq(
        model,
        example["image_path"],
        example["ambiguous_question"],
        candidates,
        n_intents=n_intents,
    )
    best_cq = uot_result["best_cq"]

    # 5. Simulate oracle-free user response to the selected CQ
    user_response = simulate_user_response(
        model,
        example["image_path"],
        example["ambiguous_question"],
        best_cq,
    )

    # 6. Generate final answer
    final_answer = generate_final_answer(
        model,
        example["image_path"],
        example["ambiguous_question"],
        best_cq,
        user_response,
    )

    return {
        "id": example["id"],
        "condition": condition.name,
        "ambiguous_question": example["ambiguous_question"],
        "candidates": candidates,
        "uot": uot_result,
        "best_cq": best_cq,
        "user_response": user_response,
        "final_answer": final_answer,
        # gold fields — only used at evaluation time
        "gold_clarification": example["gold_clarification"],
        "gold_intended_question": example["gold_intended_question"],
        "gold_answer": example["gold_answer"],
        "answers": example.get("answers", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(ALL_CONDITIONS.keys()),
        default=list(ALL_CONDITIONS.keys()),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit (needed for 7B on a 15GB GPU)",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=3,
        help="CQ candidates to generate per condition per example (default: 3)",
    )
    parser.add_argument(
        "--n_intents",
        type=int,
        default=4,
        help="Intents to generate for UoT scoring (default: 4)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=None,
        help="Path to val_annotated.jsonl (default: data/val_annotated.jsonl)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Path to images directory (default: data/images/images)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    load_kwargs = {"limit": args.limit}
    if args.jsonl_path:
        load_kwargs["jsonl_path"] = args.jsonl_path
    if args.images_dir:
        load_kwargs["images_dir"] = args.images_dir
    examples = load_intent_examples(**load_kwargs)
    print(f"Loaded {len(examples)} examples")

    model = VLMWrapper(
        model_name=args.model or DEFAULT_MODEL,
        load_in_4bit=args.load_in_4bit,
    )
    conditions = [ALL_CONDITIONS[name](model) for name in args.conditions]

    total = len(examples) * len(conditions)
    print(f"Total runs: {total}  ({len(examples)} examples × {len(conditions)} conditions)")

    with open(args.output, "w") as out_f:
        with tqdm(total=total) as pbar:
            for example in examples:
                for condition in conditions:
                    try:
                        result = run_example(
                            model, condition, example,
                            args.n_candidates, args.n_intents,
                            args.temperature, args.top_p,
                        )
                    except Exception as e:
                        result = {
                            "id": example["id"],
                            "condition": condition.name,
                            "error": str(e),
                        }
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    pbar.update(1)

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()