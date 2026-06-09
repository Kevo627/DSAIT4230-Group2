"""
Run all baseline conditions on the ClearVQA referential-ambiguity subset.
Each condition is called n_samples times per example (default 5), producing
a list of candidate CQs that the downstream scoring step will rank.

Output schema per JSONL line:
  {
    "id": str,
    "condition": str,
    "ambiguous_question": str,
    "candidate_clarifications": [
        {"clarification_question": str, "reasoning": str|null,
         "raw_output": str, "_parse_failed": bool},
        ...   # n_samples entries
    ],
    "gold_clarification": str,
    "gold_intended_question": str,
    "gold_answer": str,
    "answers": [str, ...]
  }

Usage:
    # Smoke test on 5 examples
    python scripts/run_baselines.py --limit 5

    # Run specific conditions
    python scripts/run_baselines.py --conditions standard at_cot --limit 20

    # Full run (resume-safe)
    python scripts/run_baselines.py --resume

    # Override number of candidate CQs per example
    python scripts/run_baselines.py --n_samples 3
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_referential_examples
from src.model import VLMWrapper
from src.conditions.standard import StandardCondition
from src.conditions.at import ATCondition
from src.conditions.cot import CoTCondition
from src.conditions.at_cot import ATCoTCondition

ALL_CONDITIONS = {
    "standard": StandardCondition,
    "at": ATCondition,
    "cot": CoTCondition,
    "at_cot": ATCoTCondition,
}

DEFAULT_OUTPUT = os.path.join("results", "baselines.jsonl")
DEFAULT_SAMPLED_OUTPUT = os.path.join("results", "baselines_sampled.jsonl")


def load_completed(output_path: str) -> set[tuple]:
    """Return set of (id, condition) pairs already saved."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add((r["id"], r["condition"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITION_NAMES,
        default=ALL_CONDITION_NAMES,
        help="Which conditions to run (default: all four)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of candidate CQs to generate per (example, condition) pair",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of examples (useful for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            f"Output JSONL path (default: {DEFAULT_OUTPUT}, or "
            f"{DEFAULT_SAMPLED_OUTPUT} when --sample-candidates > 1)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (id, condition) pairs already present in the output file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name or local model path (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--sample-candidates",
        type=int,
        default=1,
        help=(
            "Generate this many sampled clarification candidates per example. "
            "When > 1, select the best candidate with BERTScore against "
            "ClearVQA's clarification_question field."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used when --sample-candidates > 1",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p used when --sample-candidates > 1",
    )
    parser.add_argument(
    "--load_in_4bit",
    action="store_true",
    help="Load model in 4-bit (NF4) quantization — required for 7B on a single T4",
    )
    args = parser.parse_args()

    if args.sample_candidates < 1:
        raise ValueError("--sample-candidates must be at least 1")

    if args.output is None:
        args.output = (
            DEFAULT_SAMPLED_OUTPUT
            if args.sample_candidates > 1
            else DEFAULT_OUTPUT
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    examples = load_referential_examples(limit=args.limit)
    print(f"Loaded {len(examples)} referential-ambiguity examples")

    completed = load_completed(args.output) if args.resume else set()
    if completed:
        print(f"Resuming — {len(completed)} (id, condition) pairs already done")
    if args.sample_candidates > 1:
        print(
            "Sampling mode: generating "
            f"{args.sample_candidates} candidates per example and selecting "
            "with BERTScore against ClearVQA clarification_question"
        )

    model = VLMWrapper(model_name=args.model, load_in_4bit=args.load_in_4bit) if args.model \
    else VLMWrapper(load_in_4bit=args.load_in_4bit)
    conditions = [ALL_CONDITIONS[name](model) for name in args.conditions]
    
    """ kaggle vs local again """
    model = VLMWrapper(model_name=args.model) if args.model else VLMWrapper()
    conditions = [all_conditions[name](model) for name in args.conditions]
    

  
    total = len(examples) * len(conditions)
    skipped = sum(
        1 for ex in examples for c in conditions
        if (ex["id"], c.name) in completed
    )
    print(
        f"Total (example, condition) pairs: {total} | "
        f"To run: {total - skipped} | Skipped: {skipped} | "
        f"CQ candidates per pair: {args.n_samples}"
    )

    with open(args.output, "a") as out_f:
        with tqdm(total=total - skipped, desc="Running conditions") as pbar:
            for example in examples:
                for condition in conditions:
                    if (example["id"], condition.name) in completed:
                        continue
                    try:
                        if args.sample_candidates > 1:
                            result = condition.run_sampled(
                                example,
                                num_candidates=args.sample_candidates,
                                temperature=args.temperature,
                                top_p=args.top_p,
                            )
                            result["selection_strategy"] = "bertscore_gold_clarification"
                        else:
                            result = condition.run(example)
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
