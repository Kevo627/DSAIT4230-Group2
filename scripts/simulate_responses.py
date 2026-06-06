import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import VLMWrapper
from src.user_simulator import UserSimulator
DEFAULT_INPUT = os.path.join("results", "baselines_scored.jsonl")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    valid_rows = [r for r in rows if "generated_clarification" in r and "error" not in r]
    skipped = len(rows) - len(valid_rows)
    if skipped:
        print(f"Skipped {skipped} rows with missing 'generated_clarification' or errors")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    completed = load_completed(args.output) if args.resume else set()
    if completed:
        print(f"Resuming — {len(completed)} (id, condition) pairs already done")

    model = VLMWrapper(model_name=args.model, load_in_4bit=args.load_in_4bit) if args.model \
        else VLMWrapper(load_in_4bit=args.load_in_4bit)
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
                    clarification_question=row["generated_clarification"],
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