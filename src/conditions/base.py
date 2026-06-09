"""
Base class for all clarification conditions.
Each condition implements build_prompt() and parses model output into a result dict.
"""

from abc import ABC, abstractmethod
from src.model import VLMWrapper, parse_json_output
from src.metrics.metrics import best_bert_score_candidate


class BaseCondition(ABC):
    name: str = "base"

    def __init__(self, model: VLMWrapper):
        self.model = model

    @abstractmethod
    def build_prompt(self, ambiguous_question: str) -> str:
        pass

    def run(self, example: dict) -> dict:
        prompt = self.build_prompt(example["ambiguous_question"])
        raw_output = self.model.generate(example["image_path"], prompt)
        parsed = parse_json_output(raw_output)

        return {
            "id": example["id"],
            "condition": self.name,
            "ambiguous_question": example["ambiguous_question"],
            "generated_clarification": parsed.get("clarification_question", ""),
            "reasoning": parsed.get("reasoning", None),
            "plausible_intents": parsed.get("plausible_intents", None),
            "ambiguity_subtype": parsed.get("ambiguity_subtype", None),
            "answer_impact": parsed.get("answer_impact", None),
            "raw_output": raw_output,
            "_parse_failed": parsed.get("_parse_failed", False),
            "gold_clarification": example["gold_clarification"],
            "gold_intended_question": example["gold_intended_question"],
            "gold_answer": example["gold_answer"],
            "answers": example.get("answers", []),
        }

    def run_sampled(
        self,
        example: dict,
        num_candidates: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        scorer=best_bert_score_candidate,
        reference_key: str = "gold_clarification",
    ) -> dict:
        prompt = self.build_prompt(example["ambiguous_question"])
        parsed_outputs = []

        for _ in range(num_candidates):
            raw_output = self.model.generate(
                example["image_path"],
                prompt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            parsed = parse_json_output(raw_output)
            parsed_outputs.append({
                "generated_clarification": parsed.get("clarification_question", ""),
                "reasoning": parsed.get("reasoning", None),
                "raw_output": raw_output,
                "_parse_failed": parsed.get("_parse_failed", False),
            })

        candidates = [
            output["generated_clarification"]
            for output in parsed_outputs
        ]
        selection = scorer(candidates, example[reference_key])

        return {
            "id": example["id"],
            "condition": self.name,
            "ambiguous_question": example["ambiguous_question"],
            "generated_clarification": selection["best_candidate"],
            "selected_candidate_index": selection["best_index"],
            "selected_candidate_score": selection["best_score"],
            "candidate_scores": selection["scores"],
            "generated_candidates": parsed_outputs,
            "gold_clarification": example["gold_clarification"],
            "gold_intended_question": example["gold_intended_question"],
            "gold_answer": example["gold_answer"],
            "answers": example.get("answers", []),
        }
