"""
Base class for all clarification conditions.
Each condition implements build_prompt() and parses model output into a result dict.
"""

from abc import ABC, abstractmethod
from src.model import VLMWrapper, parse_json_output


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
            "raw_output": raw_output,
            "_parse_failed": parsed.get("_parse_failed", False),
            "gold_clarification": example["gold_clarification"],
            "gold_intended_question": example["gold_intended_question"],
            "gold_answer": example["gold_answer"],
            "answers": example.get("answers", []),
        }