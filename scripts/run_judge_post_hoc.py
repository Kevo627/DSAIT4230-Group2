import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import VLMWrapper
from src.metrics.llm_judge import llm_judge_quality

DEFAULT_OUTPUT_SUFFIX = "_judge_eval"


def _default_output(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return base + DEFAULT_OUTPUT_SUFFIX + (ext or ".jsonl")


def load_completed(output_path: str) -> set[tuple]:
    done: set[tuple] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], r["condition"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-model", default=None, dest="judge_model")
    args = parser.parse_args()

    output_path = args.output or _default_output(args.input)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row or not row.get("generated_clarification"):
                continue
            if args.conditions and row.get("condition") not in args.conditions:
                continue
            rows.append(row)

    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} rows from {args.input}")

    completed = load_completed(output_path) if args.resume else set()
    if completed:
        print(f"Resuming — {len(completed)} rows already done")

    to_run = [r for r in rows if (r["id"], r["condition"]) not in completed]
    print(f"To evaluate: {len(to_run)}")

    judge_model = VLMWrapper(model_name=args.judge_model) if args.judge_model else VLMWrapper()

    with open(output_path, "a") as out_f:
        for row in tqdm(to_run, desc="Judge eval"):
            try:
                scores = llm_judge_quality(
                    clarification_question=row["generated_clarification"],
                    ambiguous_question=row["ambiguous_question"],
                    image_path=row.get("image_path", ""),
                    model=judge_model,
                )
            except Exception as e:
                scores = {
                    "faithfulness": None,
                    "reasonableness": None,
                    "clarity": None,
                    "mean": None,
                    "raw_output": {"error": str(e)},
                }

            out_row = {
                "id": row["id"],
                "condition": row["condition"],
                "ambiguous_question": row["ambiguous_question"],
                "generated_clarification": row["generated_clarification"],
                "gold_clarification": row.get("gold_clarification", ""),
                "judge_faithfulness": scores["faithfulness"],
                "judge_reasonableness": scores["reasonableness"],
                "judge_clarity": scores["clarity"],
                "judge_mean": scores["mean"],
                "judge_raw_output": scores["raw_output"],
            }
            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()

    print(f"\nDone. Judge scores saved to {output_path}")


if __name__ == "__main__":
    main()