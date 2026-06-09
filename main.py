"""
End-to-end clarification pipeline.

Pipeline:
Dataset -> prompt condition -> candidate clarification questions ->
BERTScore selection -> simulated user response -> final answer response.
"""

# Final row fields:
# id, condition, image_path, ambiguous_question, candidate_clarifications,
# gold_clarification, gold_referential_question, gold_answer, answers,
# chosen_strategy, original_image, original_blurred_question,
# generated_clarification_questions, generated_clarification,
# selected_clarification, selected_candidate_index, selected_candidate_score,
# candidate_scores, selection_strategy, user_response, simulate_raw_output,
# simulate_parse_failed, answer_response, answer_raw_output, answer_parse_failed,
# answer_strategy.

import argparse
import json
import os
from typing import Any


CONDITION_NAMES = ["standard", "at_cot", "answer_impact"]
DEFAULT_OUTPUT = os.path.join("results", "pipeline.jsonl")
ANSWER_STRATEGY = "vlm_chat_history_selected_cq"

FINAL_ANSWER_REQUEST = """\
Now answer the original ambiguous question using the image and the clarification exchange.

Return ONLY this JSON object, no extra text:
{{
  "answer_response": "..."
}}"""

# Helper to build a text message dict for the VLM chat interface
def text_message(role: str, text: str) -> dict:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def load_completed(output_path: str) -> set[tuple[str, str]]:
    completed = set()
    if not os.path.exists(output_path):
        return completed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            example_id = row.get("id")
            condition = row.get("condition")
            if example_id is not None and condition is not None:
                completed.add((str(example_id), str(condition)))

    return completed


def build_condition(condition_name: str, model: Any):
    if condition_name == "standard":
        from src.conditions.standard import StandardCondition

        return StandardCondition(model)
    if condition_name == "at_cot":
        from src.conditions.at_cot import ATCoTCondition

        return ATCoTCondition(model)
    if condition_name == "answer_impact":
        from src.conditions.answer_impact import AnswerImpactCondition

        return AnswerImpactCondition(model)

    raise ValueError(
        f"Unknown condition '{condition_name}'. "
        f"Choose one of: {', '.join(CONDITION_NAMES)}"
    )


def select_best_clarification(
    row: dict,
    reference_key: str,
    metric: str,
) -> dict:
    if reference_key not in row:
        raise KeyError(f"Reference key '{reference_key}' not found in row")

    from src.metrics.metrics import best_bert_score_candidate

    candidates = [
        candidate.get("clarification_question", "").strip()
        for candidate in row["candidate_clarifications"]
    ]
    selection = best_bert_score_candidate(candidates, row[reference_key], metric=metric)

    return {
        **row,
        "generated_clarification": selection["best_candidate"],
        "selected_clarification": selection["best_candidate"],
        "selected_candidate_index": selection["best_index"],
        "selected_candidate_score": selection["best_score"],
        "candidate_scores": selection["scores"],
        "selection_strategy": f"bertscore_{metric}_{reference_key}",
    }


def add_question_quality_fields(row: dict) -> dict:
    clarification_questions = [
        candidate.get("clarification_question", "")
        for candidate in row.get("candidate_clarifications", [])
    ]

    return {
        **row,
        "chosen_strategy": row["condition"],
        "original_image": row["image_path"],
        "original_blurred_question": row["ambiguous_question"],
        "generated_clarification_questions": clarification_questions,
    }


def add_simulated_user_response(row: dict, simulator: Any) -> dict:
    sim = simulator.simulate(
        image_path=row["image_path"],
        ambiguous_question=row["ambiguous_question"],
        gold_referential_question=row["gold_referential_question"],
        clarification_question=row["generated_clarification"],
    )

    return {
        **row,
        "user_response": sim["user_response"],
        "simulate_raw_output": sim["raw_output"],
        "simulate_parse_failed": sim["_parse_failed"],
    }


# Build the message list for the final answer generation to simulate full conversation
def build_final_answer_messages(row: dict) -> list[dict]:
    first_turn = (
        "The user asked this ambiguous visual question:\n"
        f"{row['ambiguous_question']}\n\n"
        "Ask one clarification question that identifies which visible referent "
        "the user means. Do not answer the original question yet."
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(row['image_path'])}",
                },
                {"type": "text", "text": first_turn},
            ],
        },
        text_message("assistant", row["generated_clarification"]),
        text_message("user", row["user_response"]),
        text_message("user", FINAL_ANSWER_REQUEST),
    ]

# parse the final answer from the VLM 
def parse_answer_response(raw_output: str) -> dict:
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            answer = parsed.get(
                "answer_response",
                parsed.get("final_answer", parsed.get("answer", "")),
            )
            return {
                "answer_response": str(answer).strip(),
                "answer_parse_failed": "answer_response" not in parsed
                and "final_answer" not in parsed
                and "answer" not in parsed,
            }
    except json.JSONDecodeError:
        pass

    start = raw_output.find("{")
    if start != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw_output, start)
            if isinstance(parsed, dict):
                answer = parsed.get(
                    "answer_response",
                    parsed.get("final_answer", parsed.get("answer", "")),
                )
                return {
                    "answer_response": str(answer).strip(),
                    "answer_parse_failed": "answer_response" not in parsed
                    and "final_answer" not in parsed
                    and "answer" not in parsed,
                }
        except json.JSONDecodeError:
            pass

    return {
        "answer_response": raw_output.strip(),
        "answer_parse_failed": True,
    }

# Using the full conversation, to generate final answer, use no more than max_new_tokens (default 64) 
def generate_final_answer_response(
    row: dict,
    model: Any,
    max_new_tokens: int,
) -> dict:
    raw_output = model.generate_from_messages(
        build_final_answer_messages(row),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    parsed = parse_answer_response(raw_output)

    return {
        **row,
        "answer_response": parsed["answer_response"],
        "answer_raw_output": raw_output,
        "answer_parse_failed": parsed["answer_parse_failed"],
        "answer_strategy": ANSWER_STRATEGY,
    }


def run_pipeline_for_example(
    example: dict,
    condition: Any,
    simulator: Any,
    model: Any,
    n_samples: int,
    bert_metric: str,
    reference_key: str,
    answer_max_new_tokens: int,
) -> dict:
    row = condition.run(example, n_samples=n_samples)
    row = add_question_quality_fields(row)
    row = select_best_clarification(row, reference_key, bert_metric)
    row = add_simulated_user_response(row, simulator)
    return generate_final_answer_response(row, model, answer_max_new_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full clarification pipeline."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_NAMES,
        default=CONDITION_NAMES,
        help="Prompt conditions to run (default: all three).",
    )
    parser.add_argument(
        "--n_samples",
        "--num-candidates",
        dest="n_samples",
        type=int,
        default=5,
        help="Candidate clarification questions to generate per example.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of dataset examples.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-saved (id, condition) pairs in the output file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name or local model path.",
    )
    parser.add_argument(
        "--runtime",
        choices=["local", "kaggle"],
        default="local",
        help="Runtime preset. Kaggle enables 4-bit VLM loading by default.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the VLM in 4-bit quantization when CUDA is available.",
    )
    parser.add_argument(
        "--bert-metric",
        choices=["precision", "recall", "f1"],
        default="f1",
        help="BERTScore metric used to pick the best candidate.",
    )
    parser.add_argument(
        "--reference-key",
        choices=["gold_clarification", "gold_referential_question"],
        default="gold_clarification",
        help="Dataset field used as the BERTScore reference.",
    )
    parser.add_argument(
        "--answer-max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens for the final answer generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("--n_samples must be at least 1")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    from tqdm import tqdm
    from src.dataset import load_referential_examples
    from src.model import VLMWrapper
    from src.user_simulator import UserSimulator

    examples = load_referential_examples(limit=args.limit)
    completed = load_completed(args.output) if args.resume else set()

    print(f"Loaded {len(examples)} referential-ambiguity examples")
    print(
        f"Running {len(args.conditions)} condition(s), "
        f"{args.n_samples} clarification candidates each"
    )
    if completed:
        print(f"Resume enabled: skipping {len(completed)} completed pairs")

    load_in_4bit = args.load_in_4bit or args.runtime == "kaggle"
    model_kwargs = {"load_in_4bit": load_in_4bit}
    if args.model:
        model_kwargs["model_name"] = args.model
    model = VLMWrapper(**model_kwargs)
    simulator = UserSimulator(model)
    conditions = [
        build_condition(condition_name, model)
        for condition_name in args.conditions
    ]

    total = len(examples) * len(conditions)
    skipped = sum(
        1
        for example in examples
        for condition in conditions
        if (example["id"], condition.name) in completed
    )

    with open(args.output, "a", encoding="utf-8") as out_f:
        with tqdm(total=total - skipped, desc="Running pipeline") as pbar:
            for example in examples:
                for condition in conditions:
                    if (example["id"], condition.name) in completed:
                        continue

                    try:
                        result = run_pipeline_for_example(
                            example,
                            condition,
                            simulator,
                            model,
                            n_samples=args.n_samples,
                            bert_metric=args.bert_metric,
                            reference_key=args.reference_key,
                            answer_max_new_tokens=args.answer_max_new_tokens,
                        )
                    except Exception as exc:
                        result = {
                            "id": example["id"],
                            "condition": condition.name,
                            "error": str(exc),
                        }

                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    pbar.update(1)

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
