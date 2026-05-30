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

