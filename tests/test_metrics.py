import math
import sys
import types


bert_score_stub = types.ModuleType("bert_score")
bert_score_stub.score = lambda *args, **kwargs: None
sys.modules.setdefault("bert_score", bert_score_stub)

from src.metrics.metrics import (
    VQAScorer,
    best_bert_score_candidate,
    bert_score_comp,
    bert_score_candidates,
    distribution_comp,
    entropy_reduc,
    improvement_over_reference,
    majority_answer_change,
    mean_convo_length,
    valid_answer_prob,
    valid_probability_change,
)
import src.metrics.metrics as metrics_module


class FakeScore:
    def __init__(self, value):
        self.value = value

    def mean(self):
        return self.value



def test_vqa_normalization_and_score():
    scorer = VQAScorer()

    assert scorer.normalize("The TWO, cats!") == "2 cats"
    assert scorer.vqa_score("two", ["2", "two", "three", "two"]) == 1.0
    assert scorer.exact_match("An apple.", ["the apple", "banana"]) == 1.0


def test_distribution_and_entropy_reduction():
    scorer = VQAScorer()

    assert distribution_comp(["Yes", "yes!", "No"], scorer) == {
        "yes": 2 / 3,
        "no": 1 / 3,
    }

    result = entropy_reduc(["yes", "no", "yes"], ["yes", "yes", "yes"], scorer)

    assert result["distribution_before"] == {"yes": 2 / 3, "no": 1 / 3}
    assert result["distribution_after"] == {"yes": 1.0}
    assert math.isclose(result["entropy_before"], 0.6365141682948128)
    assert result["entropy_after"] == -0.0
    assert math.isclose(result["entropy_reduction"], 0.6365141682948128)


def test_probability_metrics():
    scorer = VQAScorer()

    assert valid_answer_prob([], ["yes"], scorer) == 0.0
    assert valid_answer_prob(["yes", "maybe"], ["yes", "no"], scorer) == 0.5

    assert valid_probability_change(
        ["yes", "maybe"],
        ["yes", "no"],
        ["yes", "no"],
        scorer,
    ) == {
        "p_valid_before": 0.5,
        "p_valid_after": 1.0,
        "delta_p_valid": 0.5,
    }


def test_majority_answer_change():
    scorer = VQAScorer()

    assert majority_answer_change(
        ["yes", "no"],
        ["yes", "yes"],
        ["yes", "yes", "no"],
        scorer,
    ) == {
        "majority_answer": "yes",
        "p_majority_before": 0.5,
        "p_majority_after": 1.0,
        "delta_p_majority": 0.5,
    }


def test_simple_summary_metrics():
    assert mean_convo_length([]) == 0.0
    assert mean_convo_length([1, 2, 3]) == 2.0
    assert improvement_over_reference(0.75, 0.5) == {
        "diff": 0.25,
        "diff_percent": 50.0,
    }
    assert improvement_over_reference(0.2, 0.0) == {
        "diff": 0.2,
        "diff_percent": None,
    }


def test_bert_score_comp_averages_score_outputs(monkeypatch):
    def fake_score(generated_questions, reference_questions, lang, verbose):
        assert generated_questions == ["what color is it?"]
        assert reference_questions == ["what is the color?"]
        assert lang == "en"
        assert verbose is False
        return FakeScore(0.8), FakeScore(0.7), FakeScore(0.75)

    monkeypatch.setattr(metrics_module, "score", fake_score)

    assert bert_score_comp(
        ["what color is it?"],
        ["what is the color?"],
    ) == {
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
    }


def test_bert_score_candidates_scores_each_candidate(monkeypatch):
    def fake_score(generated_questions, reference_questions, lang, verbose):
        assert generated_questions == [
            "what color is the car?",
            "what do you want to know about the car?",
        ]
        assert reference_questions == [
            "what color is the vehicle?",
            "what color is the vehicle?",
        ]
        return [0.7, 0.6], [0.8, 0.5], [0.75, 0.55]

    monkeypatch.setattr(metrics_module, "score", fake_score)

    assert bert_score_candidates(
        [
            "what color is the car?",
            "what do you want to know about the car?",
        ],
        "what color is the vehicle?",
    ) == [
        {
            "candidate": "what color is the car?",
            "precision": 0.7,
            "recall": 0.8,
            "f1": 0.75,
        },
        {
            "candidate": "what do you want to know about the car?",
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
        },
    ]


def test_best_bert_score_candidate_selects_highest_metric(monkeypatch):
    def fake_score(generated_questions, reference_questions, lang, verbose):
        return [0.9, 0.6], [0.4, 0.8], [0.55, 0.7]

    monkeypatch.setattr(metrics_module, "score", fake_score)

    result = best_bert_score_candidate(
        ["specific but low recall", "better overall"],
        "reference clarification",
    )

    assert result["best_candidate"] == "better overall"
    assert result["best_index"] == 1
    assert result["best_score"] == 0.7
    assert len(result["scores"]) == 2


def test_best_bert_score_candidate_handles_empty_candidates():
    assert best_bert_score_candidate([], "reference clarification") == {
        "best_candidate": "",
        "best_index": None,
        "best_score": 0.0,
        "scores": [],
    }
