import os
import zipfile
from huggingface_hub import hf_hub_download

REPO_ID = "jian0418/ClearVQA"
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSONL_PATH = os.path.join(DATA_DIR, "val_annotated.jsonl")


def download_jsonl():
    if os.path.exists(JSONL_PATH) and os.path.getsize(JSONL_PATH) > 0:
        print(f"[skip] {JSONL_PATH} already exists")
        return
    print("Downloading val_annotated.jsonl ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename="val_annotated.jsonl",
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print(f"Saved to {JSONL_PATH}")


def download_images():
    if os.path.exists(IMAGES_DIR) and any(
        f.endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(IMAGES_DIR)
    ):
        print(f"[skip] {IMAGES_DIR} already exists and is non-empty")
        return
    print("Downloading images.zip (~3.2GB, this will take a while) ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="images.zip",
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print("Extracting images ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Print first few entries so we can verify folder structure
        sample = zf.namelist()[:5]
        print(f"Zip entries (sample): {sample}")
        zf.extractall(IMAGES_DIR)
    print(f"Images extracted to {IMAGES_DIR}")


if __name__ == "__main__":
    download_jsonl()
    download_images()
    print("\nSetup complete.")
    print(f"  Annotations : {JSONL_PATH}")
    print(f"  Images      : {IMAGES_DIR}")