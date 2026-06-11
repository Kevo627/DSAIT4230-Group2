from abc import ABC, abstractmethod
from src.model import VLMWrapper, parse_json_output
from src.metrics.metrics import best_bert_score_candidate

N_SAMPLES = 5          
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
            "raw_output": raw,
            "_parse_failed": parsed.get("_parse_failed", False),
        }

    def run(self, example: dict, n_samples: int = N_SAMPLES) -> dict:
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
            "image_path": example["image_path"],   # add this line
            "ambiguous_question": example["ambiguous_question"],
            "candidate_clarifications": candidates,
            "gold_clarification": example["gold_clarification"],
            "gold_referential_question": example["gold_referential_question"],
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
                "clarification_question": parsed.get("clarification_question", ""),
                "raw_output": raw_output,
                "_parse_failed": parsed.get("_parse_failed", False),
            })

        candidates = [
            output["clarification_question"]
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
            "gold_referential_question": example["gold_referential_question"],
            "gold_answer": example["gold_answer"],
            "answers": example.get("answers", []),
        }
