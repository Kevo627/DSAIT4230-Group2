import sys
import types

class FakeVLMWrapper:
    def __init__(self, responses=None, model_name=None):
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        self.calls.append({"image_path": image_path, "prompt": prompt, **kwargs})
        if self._responses:
            return self._responses.pop(0)
        return '{"scores": [3, 4, 5]}'

    def load(self):
        pass


fake_model_mod = types.ModuleType("src.model")
fake_model_mod.VLMWrapper = FakeVLMWrapper
fake_model_mod.parse_json_output = lambda text: {}
sys.modules["src.model"] = fake_model_mod

bert_score_stub = types.ModuleType("bert_score")
bert_score_stub.score = lambda *a, **kw: None
sys.modules.setdefault("bert_score", bert_score_stub)

import json

from src.metrics.llm_judge import (
    llm_judge_candidates,
    _extract_json_field,
    _parse_score,
)


def test_parse_score_valid():
    assert _parse_score(4) == 4
    assert _parse_score("3") == 3


def test_parse_score_out_of_range():
    assert _parse_score(0) is None
    assert _parse_score(6) is None
    assert _parse_score("bad") is None


def test_extract_json_field_happy_path():
    assert _extract_json_field('{"faithfulness": 4}', "faithfulness") == 4


def test_extract_json_field_missing_key():
    assert _extract_json_field('{"other": 2}', "faithfulness") is None


def test_extract_json_field_malformed_json():
    assert _extract_json_field("not json at all", "x") is None


def test_judge_candidates_scores_all():
    response = json.dumps({"candidates": [
        {"faithfulness": 4, "reasonableness": 3, "clarity": 5},
        {"faithfulness": 2, "reasonableness": 4, "clarity": 3},
    ]})
    model = FakeVLMWrapper(responses=[response])
    result = llm_judge_candidates(
        candidates=["cq_a", "cq_b"],
        ambiguous_question="What is on the sign?",
        image_path="img.jpg",
        model=model,
    )
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["faithfulness"] == 4
    assert result["candidates"][1]["reasonableness"] == 4
    assert result["mean_clarity"] == 4.0


def test_judge_candidates_single_call():
    response = json.dumps({"candidates": [{"faithfulness": 3, "reasonableness": 3, "clarity": 3}]})
    model = FakeVLMWrapper(responses=[response])
    llm_judge_candidates(["cq"], "Q?", "img.jpg", model)
    assert len(model.calls) == 1


def test_judge_candidates_empty_input():
    model = FakeVLMWrapper()
    result = llm_judge_candidates([], "Q?", "img.jpg", model)
    assert result["candidates"] == []
    assert result["mean_faithfulness"] is None
    assert len(model.calls) == 0


def test_judge_candidates_bad_json_graceful():
    model = FakeVLMWrapper(responses=["not json"])
    result = llm_judge_candidates(["cq_a", "cq_b"], "Q?", "img.jpg", model)
    assert len(result["candidates"]) == 2
    assert all(r["faithfulness"] is None for r in result["candidates"])


def test_judge_candidates_passes_image_path():
    response = json.dumps({"candidates": [{"faithfulness": 3, "reasonableness": 3, "clarity": 3}]})
    model = FakeVLMWrapper(responses=[response])
    llm_judge_candidates(["cq"], "Q?", "data/images/test.jpg", model)
    assert model.calls[0]["image_path"] == "data/images/test.jpg"