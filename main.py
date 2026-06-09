"""
End-to-end clarification pipeline.

Pipeline:
Dataset -> prompt condition -> candidate clarification questions ->
BERTScore selection -> simulated user response -> TODO answer response.
"""

# Final row fields:
# id, condition, image_path, ambiguous_question, candidate_clarifications,
# gold_clarification, gold_referential_question, gold_answer, answers,
# chosen_strategy, original_image, original_blurred_question,
# generated_clarification_questions, generated_clarification,
# selected_clarification, selected_candidate_index, selected_candidate_score,
# candidate_scores, selection_strategy, user_response, simulate_raw_output,
# simulate_parse_failed, answer_response.

import argparse
import json
import os
from typing import Any


CONDITION_NAMES = ["standard", "at_cot", "answer_impact"]
DEFAULT_OUTPUT = os.path.join("results", "pipeline.jsonl")


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


def generate_final_answer_response(row: dict) -> dict:
    # TODO: Generate the final answer response using the image, original question,
    # selected clarification, and simulated user response.
    return {
        **row,
        "answer_response": None,
    }


def run_pipeline_for_example(
    example: dict,
    condition: Any,
    simulator: Any,
    n_samples: int,
    bert_metric: str,
    reference_key: str,
) -> dict:
    row = condition.run(example, n_samples=n_samples)
    row = add_question_quality_fields(row)
    row = select_best_clarification(row, reference_key, bert_metric)
    row = add_simulated_user_response(row, simulator)
    return generate_final_answer_response(row)


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
                            n_samples=args.n_samples,
                            bert_metric=args.bert_metric,
                            reference_key=args.reference_key,
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
