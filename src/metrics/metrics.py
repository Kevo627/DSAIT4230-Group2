import math
import string
from collections import Counter
import re
from bert_score import score

### VQA

class VQAScorer:
    
    ## Normalization
    
    def __init__(self):
        self.manual_map = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

        self.articles = {"a", "an", "the"}

        self.period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile(r"(\d)(\,)(\d)")

        self.punct = [
            ";", "/", "[", "]", '"', "{", "}",
            "(", ")", "=", "+", "\\", "_", "-",
            ">", "<", "@", "`", ",", "?", "!",
        ]
        
    def process_punctuation(self, text: str) -> str:
        out_text = text

        for p in self.punct:
            if (p + " " in text or " " + p in text) or re.search(self.comma_strip, text):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")

        out_text = self.period_strip.sub("", out_text)
        return out_text


    def process_digit_article(self, text: str) -> str:
        words = text.lower().split()
        output = []

        for word in words:
            word = self.manual_map.get(word, word)

            if word not in self.articles:
                output.append(word)

        return " ".join(output)


    def normalize(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.strip()

        text = self.process_punctuation(text)
        text = self.process_digit_article(text)

        return text
    
    
    ## Main equations
    
    def vqa_score(self, prediction: str, answers: list[str]) -> float:
        prediction = self.normalize(prediction)
        answers = [self.normalize(answer) for answer in answers]
        
        match = sum(prediction == answer for answer in answers)
        
        score = min(1.0, match/3.0)
        
        return score
    
    # In case we want to switch to a more exact measure
    def exact_match(self, prediction: str, answers: list[str]) -> float:
        prediction = self.normalize(prediction)
        answers = [self.normalize(answer) for answer in answers]

        return float(prediction in answers)

### BERTScore

def bert_score_comp(generated_questions: list[str], reference_questions: list[str],) -> dict:
    
    P, R, F1 = score(generated_questions, reference_questions, lang="en", verbose=False)
    
    return {
        "precision": float(P.mean()), 
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        }

### Improvement over baseline

def improvement_over_baseline(strategy_score: float, baseline_score: float) -> dict:
    diff = strategy_score - baseline_score
    
    if baseline_score == 0:
        diff_percent = None
    else:
        diff_percent = (diff / baseline_score) * 100

    return {
        "diff": diff,
        "diff_percent": diff_percent,
    }

### Mean conversation length

def mean_convo_length(conversation_lengths: list[int]) -> float:
    if len(conversation_lengths) == 0:
        return 0.0

    return sum(conversation_lengths) / len(conversation_lengths)

#### Now for the measures that need sampling

def distribution_comp(samples: list[str], scorer) -> dict[str, float]:
    
    normalized_samples = [scorer.normalize(sample) for sample in samples]

    counts = Counter(normalized_samples)
    total = sum(counts.values())

    if total == 0:
        return {}

    return {
        answer: count / total
        for answer, count in counts.items()
    }
    
### Entropy

def entropy(distribution: dict[str, float]):
    
    entropy = -sum(p * math.log(p) for p in distribution.values() if p > 0)
    
    return entropy


def entropy_reduc(samples_before: list[str], samples_after: list[str], scorer) -> dict:
    dist_before = distribution_comp(samples_before, scorer)
    dist_after = distribution_comp(samples_after, scorer)
    
    h_before = entropy(dist_before)
    h_after = entropy(dist_after)

    return {
        "distribution_before": dist_before,
        "distribution_after": dist_after,
        "entropy_before": h_before,
        "entropy_after": h_after,
        "entropy_reduction": h_before - h_after,
    }

### Ground truth Probability

def ground_truth_probability(samples: list[str], ground_truth: str, scorer) -> float:
    
    dist = distribution_comp(samples, scorer)
    ground_truth = scorer.normalize(ground_truth)

    return dist.get(ground_truth, 0.0)

#### Majority answer option

def majority_answer(answers: list[str], scorer):
    normalized = [scorer.normalize(answer) for answer in answers]
    
    return Counter(normalized).most_common(1)[0][0]

def majority_answer_change(samples_before, samples_after, answers, scorer) -> dict:
    target = majority_answer(answers, scorer)
    
    prob_before = ground_truth_probability(samples_before, target, scorer)
    prob_after = ground_truth_probability(samples_after, target, scorer)
    
    return {
        "majority_answer": target,
        "p_majority_before": prob_before,
        "p_majority_after": prob_after,
        "delta_p_majority": prob_after - prob_before,
    }

#### Any valid option

def valid_answer_prob(samples: list[str], answers: list[str], scorer) -> float:
    
    valid_answers = {scorer.normalize(ans) for ans in answers}

    normalized_samples = [scorer.normalize(sample) for sample in samples]
    if len(normalized_samples) == 0:
        return 0.0

    num_valid = sum(sample in valid_answers for sample in normalized_samples)

    return num_valid / len(normalized_samples)

def valid_probability_change(samples_before: list[str], samples_after: list[str], answers: list[str], scorer) -> dict:

    p_before = valid_answer_prob(samples_before, answers, scorer)

    p_after = valid_answer_prob(samples_after, answers, scorer)

    return {
        "p_valid_before": p_before,
        "p_valid_after": p_after,
        "delta_p_valid": p_after - p_before,
    }
    
