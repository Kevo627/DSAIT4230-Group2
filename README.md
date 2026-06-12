# DSAIT4230-Group2

Pipeline for generating, selecting, and simulating clarification questions for
ClearVQA referential ambiguity examples (`ambiguity_category == "refer"`).

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset and images:

```bash
python scripts/download_data.py
```

## Run The Pipeline

Local run:

```bash
python main.py --limit 5 --resume
```

Kaggle run:

```bash
python main.py \
  --runtime kaggle \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --n_samples 5 \
  --output results/pipeline.jsonl \
  --resume
```

`--runtime kaggle` enables 4-bit model loading by default. You can also pass
`--load_in_4bit` directly when running on a local CUDA GPU with limited memory.

## Conditions

`main.py` supports three prompting strategies:

```text
standard
at_cot
answer_impact
```

Run a single strategy:

```bash
python main.py --conditions answer_impact --limit 10
```

## Output

The default output is:

```text
results/pipeline.jsonl
```

Each JSONL row includes:

```text
id
condition
image_path
ambiguous_question
candidate_clarifications
generated_clarification
selected_candidate_index
selected_candidate_score
candidate_scores
chosen_strategy
original_image
original_blurred_question
generated_clarification_questions
user_response
answer_response
gold_clarification
gold_referential_question
gold_answer
answers
```

The `answer_response` field is currently left as `null`.

## Referential Dataset

The dataset loader uses only the referential subset:

```python
from src.dataset import load_referential_examples

examples = load_referential_examples(limit=5)
```

## Tests

```bash
pytest
```

If `bert-score` is missing in a fresh environment:

```bash
pip install bert-score
```

## LLM-as-Judge Evaluation

After running the pipeline, evaluate clarification candidates with the LLM judge:

```bash
python scripts/run_judge_post_hoc.py \
    --input results/pipeline.jsonl \
    --output results/pipeline_judge.jsonl \
    --judge-model Qwen/Qwen2.5-VL-7B-Instruct
```

Scores two dimensions per candidate (faithfulness, reasonableness) on a 1–5 scale, following G-Eval (Liu et al., 2023). Use `--resume` to continue interrupted runs.

## Repo Structure

```text
DSAIT4230-Group2/
  main.py
  scripts/
    download_data.py
    simulate_responses.py
  src/
    dataset.py
    model.py
    user_simulator.py
    conditions/
      base.py
      standard.py
      at_cot.py
      answer_impact.py
      old/
        at.py
        cot.py
        at_subtype.py
  data/
    val_annotated.jsonl
    images/
  results/
  requirements.txt
  README.md
```
