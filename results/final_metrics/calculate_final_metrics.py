import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

from scipy import stats


SPLITS = {
    "standard": {
        "judge": "LLM-as-a-Judge/judge_standard_aligned.jsonl",
        "pipeline": "full_pipeline/standard.jsonl",
    },
    "novel": {
        "judge": "LLM-as-a-Judge/judge_novel_aligned.jsonl",
        "pipeline": "full_pipeline/novel.jsonl",
    },
    "at_cot": {
        "judge": "LLM-as-a-Judge/judge_at_cot_aligned.jsonl",
        "pipeline": "full_pipeline/at_cot.jsonl",
    },
}


class VQAScorer:
    """VQA-style soft answer scorer with answer normalization."""

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
        self.period_strip = re.compile(r"(?<!\d)\.(?!\d)")
        self.comma_strip = re.compile(r"(\d),(\d)")
        self.punct = [
            ";",
            "/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
            ":",
        ]

    def process_punctuation(self, text):
        out_text = text
        for punct in self.punct:
            if (punct + " " in text or " " + punct in text) or re.search(self.comma_strip, text):
                out_text = out_text.replace(punct, "")
            else:
                out_text = out_text.replace(punct, " ")
        return self.period_strip.sub("", out_text)

    def process_digit_article(self, text):
        output = []
        for word in text.lower().split():
            word = self.manual_map.get(word, word)
            if word not in self.articles:
                output.append(word)
        return " ".join(output)

    def normalize(self, text):
        text = str(text or "").replace("\n", " ").replace("\t", " ").strip()
        text = self.process_punctuation(text)
        return self.process_digit_article(text)

    def vqa_score(self, prediction, answers):
        """Soft VQA accuracy: min(1, number of matching answers / 3)."""
        prediction = self.normalize(prediction)
        normalized_answers = [self.normalize(answer) for answer in answers if str(answer or "").strip()]

        matches = 0
        for answer in normalized_answers:
            if answer and re.search(r"\b" + re.escape(answer) + r"\b", prediction):
                matches += 1

        return min(1.0, matches / 3.0)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def numeric_values_by_id(rows, field):
    values = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        values[row["id"]] = float(value)
    return values


def ground_truth_answers(row):
    answers = []

    gold_answer = row.get("gold_answer")
    if gold_answer not in (None, ""):
        answers.append(str(gold_answer))

    for answer in row.get("answers") or []:
        if answer not in (None, ""):
            answers.append(str(answer))

    return answers


def split_scores(paths, results_dir, scorer):
    judge_rows = read_jsonl(results_dir / paths["judge"])
    pipeline_rows = read_jsonl(results_dir / paths["pipeline"])

    faithfulness = numeric_values_by_id(judge_rows, "judge_mean_faithfulness")
    reasonableness = numeric_values_by_id(judge_rows, "judge_mean_reasonableness")

    vqa_soft_accuracy = {}
    for row in pipeline_rows:
        prediction = row.get("answer_response")
        answers = ground_truth_answers(row)
        if not prediction or not answers:
            continue
        vqa_soft_accuracy[row["id"]] = scorer.vqa_score(prediction, answers)

    return {
        "mean_faithfulness": faithfulness,
        "mean_reasonableness": reasonableness,
        "vqa_soft_accuracy": vqa_soft_accuracy,
    }


def summarize_split(split, scores):
    return {
        "split": split,
        "mean_faithfulness": mean(scores["mean_faithfulness"].values()),
        "mean_reasonableness": mean(scores["mean_reasonableness"].values()),
        "vqa_soft_accuracy": mean(scores["vqa_soft_accuracy"].values()),
    }


def paired_t_tests(scores_by_split):
    tests = []
    split_names = list(SPLITS)
    metric_names = ["mean_faithfulness", "mean_reasonableness", "vqa_soft_accuracy"]

    for metric in metric_names:
        for i, split_a in enumerate(split_names):
            for split_b in split_names[i + 1 :]:
                scores_a = scores_by_split[split_a][metric]
                scores_b = scores_by_split[split_b][metric]
                shared_ids = sorted(set(scores_a) & set(scores_b))
                values_a = [scores_a[item_id] for item_id in shared_ids]
                values_b = [scores_b[item_id] for item_id in shared_ids]
                statistic, p_value = stats.ttest_rel(values_a, values_b)

                tests.append(
                    {
                        "metric": metric,
                        "comparison": f"{split_a}_vs_{split_b}",
                        "split_a": split_a,
                        "split_b": split_b,
                        "mean_a": mean(values_a),
                        "mean_b": mean(values_b),
                        "mean_difference_a_minus_b": mean(a - b for a, b in zip(values_a, values_b)),
                        "p_value": float(p_value),
                        "t_statistic": float(statistic),
                        "n": len(shared_ids),
                        "significant_at_0.05": bool(p_value < 0.05),
                    }
                )

    return tests


def write_outputs(metrics, paired_tests, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "final_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "paired_t_tests": paired_tests,
            },
            f,
            indent=2,
        )
        f.write("\n")

    csv_path = output_dir / "final_metrics.csv"
    fieldnames = [
        "split",
        "mean_faithfulness",
        "mean_reasonableness",
        "vqa_soft_accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    tests_csv_path = output_dir / "paired_t_tests.csv"
    test_fieldnames = [
        "metric",
        "comparison",
        "split_a",
        "split_b",
        "mean_a",
        "mean_b",
        "mean_difference_a_minus_b",
        "p_value",
        "t_statistic",
        "n",
        "significant_at_0.05",
    ]
    with tests_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_fieldnames)
        writer.writeheader()
        writer.writerows(paired_tests)

    return json_path, csv_path, tests_csv_path


def print_table(metrics):
    headers = ["split", "faithfulness", "reasonableness", "vqa_soft_accuracy"]
    widths = [12, 14, 16, 18]
    print("".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("".join("-" * width for width in widths))
    for row in metrics:
        values = [
            row["split"],
            f"{row['mean_faithfulness']:.4f}" if row["mean_faithfulness"] is not None else "NA",
            f"{row['mean_reasonableness']:.4f}" if row["mean_reasonableness"] is not None else "NA",
            f"{row['vqa_soft_accuracy']:.4f}" if row["vqa_soft_accuracy"] is not None else "NA",
        ]
        print("".join(value.ljust(width) for value, width in zip(values, widths)))


def main():
    parser = argparse.ArgumentParser(
        description="Compute final faithfulness, reasonableness, and VQA soft accuracy metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where final_metrics.json and final_metrics.csv are written.",
    )
    args = parser.parse_args()

    scorer = VQAScorer()
    scores_by_split = {
        split: split_scores(paths, args.results_dir, scorer)
        for split, paths in SPLITS.items()
    }
    metrics = [summarize_split(split, scores_by_split[split]) for split in SPLITS]
    paired_tests = paired_t_tests(scores_by_split)

    json_path, csv_path, tests_csv_path = write_outputs(metrics, paired_tests, args.output_dir)
    print_table(metrics)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tests_csv_path}")


if __name__ == "__main__":
    main()
