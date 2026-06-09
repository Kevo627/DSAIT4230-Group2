# DSAIT4230-Group2

# DSAIT4230-Group2

## Setup (Kaggle)

**Cell 1 — clone and install**
```python
%cd /kaggle/working
!rm -rf DSAIT4230-Group2
!git clone https://github.com/Kevo627/DSAIT4230-Group2.git
%cd DSAIT4230-Group2

!pip install transformers torch torchvision torchaudio qwen-vl-utils \
    bitsandbytes bert-score tqdm huggingface_hub pandas Pillow datasets -q

import os, transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers.logging.set_verbosity_error()
```

**Cell 2 — HF token**
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("hf-token")
```

**Cell 3 — download data (images ~3.2GB, run once)**
```python
!python scripts/download_data.py
```

**Cell 4 — run baselines**
```python
!python scripts/run_baselines.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --load_in_4bit \
    --n_samples 5 \
    --output results/baselines.jsonl \
    --resume
```

**Cell 5 — simulate user responses **
```python
!python scripts/simulate_responses.py \
    --load_in_4bit \
    --resume
```

### 4. For running the metric tests, run the following command to install bert_score. 

``pip install bert_score``
### 4. Baseline model
The default baseline model is `Qwen/Qwen2.5-VL-7B-Instruct`, loaded in 4-bit on CUDA when available. Override it with `--model` if you want to compare against another Hugging Face checkpoint or local model path.

The repo now targets Windows only, so the install uses `bitsandbytes` instead of AWQ/TRITON.

## Repo structure

DSAIT4230-Group2/
├── scripts/
│   ├── download_data.py
│   ├── run_baselines.py
│   └── simulate_responses.py
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── user_simulator.py
│   └── conditions/
│       ├── __init__.py
│       ├── base.py
│       ├── standard.py
│       ├── cot.py
│       ├── at.py
│       └── at_cot.py
├── data/
│   ├── val_annotated.jsonl
│   └── images/
│       └── images/
├── results/
├── requirements.txt
└── README.md

