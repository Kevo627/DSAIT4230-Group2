import os
import json
import pandas as pd
from PIL import Image
from typing import Optional

JSONL_PATH = os.path.join("data", "val_annotated.jsonl")
IMAGES_DIR = os.path.join("data", "images", "images")

REFERENTIAL_CATEGORY = "referential"


def load_referential_examples(
    jsonl_path: str = JSONL_PATH,
    images_dir: str = IMAGES_DIR,
    limit: Optional[int] = None,
) -> list[dict]:
    df = pd.read_json(jsonl_path, lines=True)

    if "ambiguity_category" not in df.columns:
        raise ValueError(
            "ambiguity_category column not found. "
            "Make sure you loaded val_annotated.jsonl, not the train split."
        )

    ref_df = df[df["ambiguity_category"] == REFERENTIAL_CATEGORY].reset_index(drop=True)

    if len(ref_df) == 0:
        raise ValueError(
            f"No examples found for category '{REFERENTIAL_CATEGORY}'. "
            f"Available categories: {df['ambiguity_category'].unique().tolist()}"
        )

    if limit is not None:
        ref_df = ref_df.head(limit)

    examples = []
    for _, row in ref_df.iterrows():
        image_path = os.path.join(images_dir, row["image"])
        examples.append({
            "id": str(row["question_id"]),
            "image_path": image_path,
            "ambiguous_question": row["blurred_question"],
            "gold_intended_question": row["question"],
            "gold_clarification": row["clarification_question"],
            "gold_answer": str(row["gold_answer"]),
            "answers": [str(a) for a in row.get("answers", [])],
        })

    return examples


def load_image(image_path: str) -> Image.Image:
    """Load a PIL image from path. Call this lazily — don't preload everything."""
    return Image.open(image_path).convert("RGB")


def inspect_categories(jsonl_path: str = JSONL_PATH):
    """Run this once after download to confirm the category strings."""
    df = pd.read_json(jsonl_path, lines=True)
    print("Columns:", df.columns.tolist())
    print("\nambiguity_category value counts:")
    print(df["ambiguity_category"].value_counts())
    print(f"\nTotal examples: {len(df)}")


if __name__ == "__main__":
    inspect_categories()
    print("\nLoading referential ambiguity examples ...")
    examples = load_referential_examples(limit=3)
    print(f"Loaded {len(examples)} examples (showing first 3)\n")
    for ex in examples:
        print(f"ID                   : {ex['id']}")
        print(f"Ambiguous Q          : {ex['ambiguous_question']}")
        print(f"Gold intended Q      : {ex['gold_intended_question']}")
        print(f"Gold clarification   : {ex['gold_clarification']}")
        print(f"Gold answer          : {ex['gold_answer']}")
        print(f"Image path           : {ex['image_path']}")
        print(f"Image exists         : {os.path.exists(ex['image_path'])}")
        print()
