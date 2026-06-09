"""
Select one clarification question from independently generated candidates.

This stage intentionally uses a deterministic placeholder scorer. It preserves
the pipeline shape while keeping real external scorers, BERTScore, or
LLM-as-a-judge integrations out of this pass.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_INPUT = os.path.join("results", "baselines.jsonl")
DEFAULT_OUTPUT = os.path.join("results", "baselines_selected.jsonl")
SELECTION_STRATEGY = "placeholder_first_valid"


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


def candidate_text(candidate: dict) -> str:
    return str(candidate.get("clarification_question", "")).strip()


def placeholder_select(candidates: list[dict]) -> dict:
    selected_index = None
    selected_text = ""

    for index, candidate in enumerate(candidates):
        text = candidate_text(candidate)
        if text and not candidate.get("_parse_failed", False):
            selected_index = index
            selected_text = text
            break

    if selected_index is None:
        for index, candidate in enumerate(candidates):
            text = candidate_text(candidate)
            if text:
                selected_index = index
                selected_text = text
                break

    scores = []
    for index, candidate in enumerate(candidates):
        text = candidate_text(candidate)
        scores.append({
            "candidate": text,
            "score": 1.0 if index == selected_index else 0.0,
            "is_placeholder_score": True,
        })

    return {
        "generated_clarification": selected_text,
        "selected_candidate_index": selected_index,
        "selected_candidate_score": 1.0 if selected_index is not None else 0.0,
        "candidate_scores": scores,
        "selection_strategy": SELECTION_STRATEGY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    valid_rows = [
        r for r in rows
        if "candidate_clarifications" in r and "error" not in r
    ]
    skipped = len(rows) - len(valid_rows)
    if skipped:
        print(f"Skipped {skipped} rows with missing candidates or errors")

    if args.limit is not None:
        valid_rows = valid_rows[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    completed = load_completed(args.output) if args.resume else set()
    if completed:
        print(f"Resuming - {len(completed)} (id, condition) pairs already done")

    to_run = [
        r for r in valid_rows
        if (r["id"], r["condition"]) not in completed
    ]
    print(f"Total valid rows: {len(valid_rows)} | To select: {len(to_run)}")

    with open(args.output, "a") as out_f:
        for row in tqdm(to_run, desc="Selecting clarifications"):
            selection = placeholder_select(row["candidate_clarifications"])
            out_row = {**row, **selection}
            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
