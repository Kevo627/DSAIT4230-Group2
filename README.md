# DSAIT4230-Group2

## Setup

### 0. Venv setup
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -c "import torch; print(torch.backends.mps.is_available())"

python scripts/run_baselines.py --limit 2 --conditions standard
### 1. Clone and install dependencies
pip install -r requirements.txt

### 2. Download dataset (images ~3.2GB, run once) - first 3 steps optional, needed only for faster download
- huggingface.co → log in (or create a free account)
- Settings → Access Tokens → New token → read permissions → copy it
- Run: `hf auth login` and paste the token in CLI
- python scripts/download_data.py

### 3. Verify the dataset loaded correctly
python src/dataset.py

### 4. For running the metric tests, run the following command to install bert_score. 

``pip install bert_score``
### 4. Baseline model
The default baseline model is `Qwen/Qwen2.5-VL-7B-Instruct`, loaded in 4-bit on CUDA when available. Override it with `--model` if you want to compare against another Hugging Face checkpoint or local model path.

The repo now targets Windows only, so the install uses `bitsandbytes` instead of AWQ/TRITON.

## Repo structure

├── scripts/
│   └── download_data.py        # One-time dataset download
├── src/
│   ├── dataset.py              # Dataset loader and filter
│   ├── conditions/             # One file per prompting condition
│   │   ├── standard.py
│   │   ├── at.py
│   │   ├── cot.py
│   │   ├── at_cot.py
│   │   └── subtype_guided.py
│   └── evaluate/               # Metrics
├── data/                       # gitignored — created by download_data.py
│   ├── val_annotated.jsonl
│   └── images/
├── results/                    # gitignored — experiment outputs
├── requirements.txt
└── README.md


NOTE: When downloading, training images are downloaded too, which took up unnecessary space. I deleted those using this on MacOS: 
find data/images/images -name "train_*.jpg" | wc -l
find data/images/images -name "train_*.jpg" -delete

