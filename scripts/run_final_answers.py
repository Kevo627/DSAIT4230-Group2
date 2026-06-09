"""
Answer the original ambiguous question after clarification.

This stage uses the real VLMWrapper, but the prompt is kept leakage-free: it
includes only the image, original ambiguous question, selected clarification
question, and simulated user response.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import VLMWrapper

DEFAULT_INPUT = os.path.join("results", "with_user_responses.jsonl")
DEFAULT_OUTPUT = os.path.join("results", "final_answers.jsonl")
FINAL_ANSWER_STRATEGY = "vlm_observed_clarification_context"

FINAL_ANSWER_PROMPT = """\
You are answering a visual question after a clarification exchange.

Original ambiguous question: "{ambiguous_question}"
Clarification question: "{clarification_question}"
User response to the clarification question: "{user_response}"

Use the image and the clarification exchange to answer the original question.
Return ONLY a JSON object, no extra text:
{{
    "final_answer": "..."
}}"""


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


def build_prompt(row: dict) -> str:
    return FINAL_ANSWER_PROMPT.format(
        ambiguous_question=row["ambiguous_question"],
        clarification_question=row["generated_clarification"],
        user_response=row["user_response"],
    )


def parse_final_answer(text: str) -> dict:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            answer = parsed.get("final_answer", parsed.get("answer", ""))
            return {
                "final_answer": str(answer).strip(),
                "final_parse_failed": "final_answer" not in parsed and "answer" not in parsed,
            }
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                answer = obj.get("final_answer", obj.get("answer", ""))
                return {
                    "final_answer": str(answer).strip(),
                    "final_parse_failed": "final_answer" not in obj and "answer" not in obj,
                }
        except json.JSONDecodeError:
            pass

    return {
        "final_answer": text.strip(),
        "final_parse_failed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    valid_rows = [
        r for r in rows
        if "error" not in r
        and "simulate_error" not in r
        and r.get("image_path")
        and r.get("ambiguous_question")
        and r.get("generated_clarification")
        and r.get("user_response")
    ]
    skipped = len(rows) - len(valid_rows)
    if skipped:
        print(f"Skipped {skipped} rows missing final-answer inputs or containing errors")

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
    print(f"Total valid rows: {len(valid_rows)} | To answer: {len(to_run)}")

    model = VLMWrapper(model_name=args.model, load_in_4bit=args.load_in_4bit) if args.model \
        else VLMWrapper(load_in_4bit=args.load_in_4bit)

    with open(args.output, "a") as out_f:
        for row in tqdm(to_run, desc="Generating final answers"):
            try:
                prompt = build_prompt(row)
                raw_output = model.generate(
                    row["image_path"],
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                parsed = parse_final_answer(raw_output)
                out_row = {
                    **row,
                    "final_answer": parsed["final_answer"],
                    "final_raw_output": raw_output,
                    "final_parse_failed": parsed["final_parse_failed"],
                    "final_answer_strategy": FINAL_ANSWER_STRATEGY,
                }
            except Exception as e:
                out_row = {**row, "final_answer_error": str(e)}

            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
