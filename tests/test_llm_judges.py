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

from src.metrics.llm_judge import (
    llm_judge_candidates,
    llm_judge_quality,
    _extract_score,
    _extract_json_field,
)


def test_extract_score_finds_valid_integer():
    assert _extract_score("The score is 4.") == 4
    assert _extract_score("I give it a 2 out of 5") == 2


def test_extract_score_ignores_out_of_range():
    assert _extract_score("score: 9") is None
    assert _extract_score("no digits here") is None


def test_extract_json_field_happy_path():
    assert _extract_json_field('Some text {"faithfulness": 4} trailing', "faithfulness") == 4


def test_extract_json_field_missing_key():
    assert _extract_json_field('{"other": 2}', "faithfulness") is None


def test_extract_json_field_malformed_json():
    assert _extract_json_field("not json at all", "x") is None


def test_judge_candidates_picks_highest_score():
    model = FakeVLMWrapper(responses=['{"scores": [2, 5, 3]}'])
    result = llm_judge_candidates(
        candidates=["cq_a", "cq_b", "cq_c"],
        reference_question="gold",
        image_path="img.jpg",
        ambiguous_question="What is on this?",
        model=model,
    )
    assert result["best_candidate"] == "cq_b"
    assert result["best_index"] == 1
    assert result["best_score"] == 5.0
    assert len(result["scores"]) == 3


def test_judge_candidates_empty_input():
    model = FakeVLMWrapper()
    result = llm_judge_candidates(
        candidates=[],
        reference_question="gold",
        image_path="img.jpg",
        ambiguous_question="What is this?",
        model=model,
    )
    assert result["best_candidate"] == ""
    assert result["best_index"] is None
    assert result["best_score"] == 0.0
    assert len(model.calls) == 0


def test_judge_candidates_fallback_on_bad_json():
    model = FakeVLMWrapper(responses=["I cannot score these."])
    result = llm_judge_candidates(
        candidates=["a", "b"],
        reference_question="gold",
        image_path="img.jpg",
        ambiguous_question="Q?",
        model=model,
    )
    assert all(s["score"] == 1.0 for s in result["scores"])


def test_judge_candidates_passes_image_to_model():
    model = FakeVLMWrapper(responses=['{"scores": [4]}'])
    llm_judge_candidates(
        candidates=["only one"],
        reference_question="ref",
        image_path="data/images/test.jpg",
        ambiguous_question="Q?",
        model=model,
    )
    assert model.calls[0]["image_path"] == "data/images/test.jpg"


def test_judge_candidates_reference_question_is_ignored():
    model = FakeVLMWrapper(responses=['{"scores": [3]}'])
    llm_judge_candidates(
        candidates=["cq"],
        reference_question="THIS_GOLD_SHOULD_NOT_APPEAR",
        image_path="img.jpg",
        ambiguous_question="Q?",
        model=model,
    )
    assert "THIS_GOLD_SHOULD_NOT_APPEAR" not in model.calls[0]["prompt"]


def test_judge_quality_returns_all_dimensions():
    model = FakeVLMWrapper(responses=[
        '{"faithfulness": 5, "reasonableness": 4, "clarity": 3}',
    ])
    result = llm_judge_quality(
        clarification_question="Which sign do you mean?",
        ambiguous_question="What is on the sign?",
        image_path="img.jpg",
        model=model,
    )
    assert result["faithfulness"] == 5
    assert result["reasonableness"] == 4
    assert result["clarity"] == 3
    assert abs(result["mean"] - 4.0) < 1e-6


def test_judge_quality_single_call():
    model = FakeVLMWrapper(responses=[
        '{"faithfulness": 5, "reasonableness": 4, "clarity": 3}',
    ])
    llm_judge_quality("CQ", "Q", "img.jpg", model)
    assert len(model.calls) == 1


def test_judge_quality_handles_partial_parse_failure():
    model = FakeVLMWrapper(responses=[
        '{"faithfulness": 4, "reasonableness": 2, "clarity": 9}',
    ])
    result = llm_judge_quality("CQ", "Q", "img.jpg", model)
    assert result["faithfulness"] == 4
    assert result["reasonableness"] == 2
    assert result["clarity"] is None
    assert abs(result["mean"] - 3.0) < 1e-6


def test_judge_quality_all_fail_returns_none_mean():
    model = FakeVLMWrapper(responses=["nope"])
    result = llm_judge_quality("CQ", "Q", "img.jpg", model)
    assert result["mean"] is None


def test_judge_quality_passes_image_path():
    model = FakeVLMWrapper(responses=[
        '{"faithfulness": 3, "reasonableness": 3, "clarity": 3}',
    ])
    llm_judge_quality("CQ", "Q", "data/images/special.jpg", model)
    assert model.calls[0]["image_path"] == "data/images/special.jpg"