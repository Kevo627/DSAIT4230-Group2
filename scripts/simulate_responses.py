"""
Simulate user responses to selected clarification questions.

This script runs AFTER the scoring step has selected the best CQ per
(example, condition) pair. It reads a JSONL file where each line has a
'selected_clarification' field (the winning CQ string), and produces a
new JSONL with a 'user_response' field appended.

Expected input schema (one line per (example, condition)):
  {
    "id": str,
    "condition": str,
    "ambiguous_question": str,
    "image_path": str,
    "gold_intended_question": str,
    "gold_answer": str,
    "gold_clarification": str,
    "selected_clarification": str   ← set by the scoring step
  }

Output adds:
  {
    "user_response": str,
    "simulate_raw_output": str,
    "simulate_parse_failed": bool
  }

Usage:
    python scripts/simulate_responses.py \
        --input results/selected_cqs.jsonl \
        --output results/with_user_responses.jsonl

    # Resume interrupted run
    python scripts/simulate_responses.py --input ... --output ... --resume
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import VLMWrapper
from src.user_simulator import UserSimulator

DEFAULT_INPUT = os.path.join("results", "selected_cqs.jsonl")
DEFAULT_OUTPUT = os.path.join("results", "with_user_responses.jsonl")


def load_completed(output_path: str) -> set[tuple]:
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
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (id, condition) pairs already present in the output file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (default: Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Filter out rows that errored in the scoring step or have no selected CQ
    valid_rows = [r for r in rows if "selected_clarification" in r and "error" not in r]
    skipped_invalid = len(rows) - len(valid_rows)
    if skipped_invalid:
        print(f"Skipped {skipped_invalid} rows with missing 'selected_clarification' or errors")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    completed = load_completed(args.output) if args.resume else set()
    if completed:
        print(f"Resuming — {len(completed)} (id, condition) pairs already done")

    model = VLMWrapper(model_name=args.model) if args.model else VLMWrapper()
    simulator = UserSimulator(model)

    to_run = [r for r in valid_rows if (r["id"], r["condition"]) not in completed]
    print(f"Total valid rows: {len(valid_rows)} | To simulate: {len(to_run)}")

    with open(args.output, "a") as out_f:
        for row in tqdm(to_run, desc="Simulating user responses"):
            try:
                sim = simulator.simulate(
                    image_path=row["image_path"],
                    ambiguous_question=row["ambiguous_question"],
                    gold_intended_question=row["gold_intended_question"],
                    clarification_question=row["selected_clarification"],
                )
                out_row = {
                    **row,
                    "user_response": sim["user_response"],
                    "simulate_raw_output": sim["raw_output"],
                    "simulate_parse_failed": sim["_parse_failed"],
                }
            except Exception as e:
                out_row = {**row, "simulate_error": str(e)}

            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
