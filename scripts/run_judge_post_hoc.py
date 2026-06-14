import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import VLMWrapper
from src.metrics.llm_judge import llm_judge_candidates

DEFAULT_OUTPUT_SUFFIX = "_judge_eval"


def _default_output(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return base + DEFAULT_OUTPUT_SUFFIX + ext


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
    parser.add_argument("--load_in_4bit", action="store_true")
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
            if "error" in row:
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

    judge_model = VLMWrapper(model_name=args.judge_model, load_in_4bit=args.load_in_4bit) if args.judge_model else VLMWrapper(load_in_4bit=args.load_in_4bit)

    with open(output_path, "a") as out_f:
        for row in tqdm(to_run, desc="Judge eval"):
            try:
                raw_candidates = row.get("candidate_clarifications", [])
                candidates = [
                    c.get("clarification_question", "")
                    for c in raw_candidates
                    if c.get("clarification_question")
                ]
                if not candidates and row.get("generated_clarification"):
                    print(f"Warning: no candidate list for id={row['id']} condition={row['condition']} — falling back to single best candidate")
                    candidates = [row["generated_clarification"]]

                image_path = row.get("image_path", "")
                if not image_path:
                    print(f"Warning: missing image_path for id={row['id']} condition={row['condition']} — Faithfulness scores will be unreliable")

                judgment = llm_judge_candidates(
                    candidates=candidates,
                    ambiguous_question=row["ambiguous_question"],
                    image_path=image_path,
                    model=judge_model,
                )
            except Exception as e:
                judgment = {
                    "candidates": [],
                    "mean_faithfulness": None,
                    "mean_reasonableness": None,
                    "raw_outputs": {"error": str(e)},
                }

            out_row = {
                "id": row["id"],
                "condition": row["condition"],
                "ambiguous_question": row["ambiguous_question"],
                "image_path": row.get("image_path", ""),
                "gold_clarification": row.get("gold_clarification", ""),
                "judge_candidates": judgment["candidates"],
                "judge_mean_faithfulness": judgment["mean_faithfulness"],
                "judge_mean_reasonableness": judgment["mean_reasonableness"],
                "judge_raw_outputs": judgment["raw_outputs"],
            }
            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()

    print(f"\nDone. Judge scores saved to {output_path}")


if __name__ == "__main__":
    main()