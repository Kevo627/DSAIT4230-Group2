import sys
import types


bert_score_stub = types.ModuleType("bert_score")
bert_score_stub.score = lambda *args, **kwargs: None
sys.modules.setdefault("bert_score", bert_score_stub)

qwen_stub = types.ModuleType("qwen_vl_utils")
qwen_stub.process_vision_info = lambda messages: (None, None)
sys.modules.setdefault("qwen_vl_utils", qwen_stub)

from src.conditions.base import BaseCondition


class DummyModel:
    def __init__(self):
        self.outputs = [
            '{"clarification_question": "what color is it?"}',
            '{"clarification_question": "what object should I inspect?"}',
        ]
        self.calls = []

    def generate(self, image_path, prompt, **kwargs):
        self.calls.append((image_path, prompt, kwargs))
        return self.outputs[len(self.calls) - 1]


class DummyCondition(BaseCondition):
    name = "dummy"

    def build_prompt(self, ambiguous_question: str) -> str:
        return f"Prompt: {ambiguous_question}"


def test_run_sampled_selects_best_candidate_with_scorer():
    model = DummyModel()
    condition = DummyCondition(model)
    example = {
        "id": "1",
        "image_path": "image.jpg",
        "ambiguous_question": "what is it?",
        "gold_clarification": "what color is the object?",
        "gold_intended_question": "what color is the object?",
        "gold_answer": "red",
        "answers": ["red"],
    }

    def fake_scorer(candidates, reference_question):
        assert candidates == [
            "what color is it?",
            "what object should I inspect?",
        ]
        assert reference_question == "what color is the object?"
        return {
            "best_candidate": "what color is it?",
            "best_index": 0,
            "best_score": 0.9,
            "scores": [
                {"candidate": "what color is it?", "f1": 0.9},
                {"candidate": "what object should I inspect?", "f1": 0.4},
            ],
        }

    result = condition.run_sampled(
        example,
        num_candidates=2,
        scorer=fake_scorer,
    )

    assert result["generated_clarification"] == "what color is it?"
    assert result["selected_candidate_index"] == 0
    assert result["selected_candidate_score"] == 0.9
    assert len(result["generated_candidates"]) == 2
    assert model.calls[0][2] == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
    }
