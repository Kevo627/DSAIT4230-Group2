"""
Base class for all clarification conditions.
Each condition implements build_prompt() and parses model output into a result dict.
"""

from abc import ABC, abstractmethod
from src.model import VLMWrapper, parse_json_output

N_SAMPLES = 5          # candidate CQs to generate per condition
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_P = 0.95


class BaseCondition(ABC):
    name: str = "base"

    def __init__(self, model: VLMWrapper):
        self.model = model

    @abstractmethod
    def build_prompt(self, ambiguous_question: str) -> str:
        pass

    def _call_once(
        self,
        image_path: str,
        prompt: str,
        do_sample: bool = False,
    ) -> dict:
        """Single model call → parsed dict with at least 'clarification_question'."""
        raw = self.model.generate(
            image_path,
            prompt,
            do_sample=do_sample,
            temperature=SAMPLE_TEMPERATURE if do_sample else 1.0,
            top_p=SAMPLE_TOP_P if do_sample else 1.0,
        )
        parsed = parse_json_output(raw)
        return {
            "clarification_question": parsed.get("clarification_question", ""),
            "reasoning": parsed.get("reasoning", None),
            "raw_output": raw,
            "_parse_failed": parsed.get("_parse_failed", False),
        }

    def run(self, example: dict, n_samples: int = N_SAMPLES) -> dict:
        """
        Generate `n_samples` candidate CQs for one example.

        The first sample is drawn greedily (do_sample=False) for reproducibility;
        the remaining n-1 are sampled for diversity.

        Returns a result dict with:
          - candidate_clarifications: list of n_samples dicts, each with
              'clarification_question', 'reasoning', 'raw_output', '_parse_failed'
          - plus metadata fields (id, condition, gold_*, …)
        """
        prompt = self.build_prompt(example["ambiguous_question"])

        candidates = []
        for i in range(n_samples):
            sample = self._call_once(
                example["image_path"],
                prompt,
                do_sample=(i > 0),   # greedy for first, sampled for rest
            )
            candidates.append(sample)

        return {
            "id": example["id"],
            "condition": self.name,
            "ambiguous_question": example["ambiguous_question"],
            "candidate_clarifications": candidates,
            "gold_clarification": example["gold_clarification"],
            "gold_intended_question": example["gold_intended_question"],
            "gold_answer": example["gold_answer"],
            "answers": example.get("answers", []),
        }
