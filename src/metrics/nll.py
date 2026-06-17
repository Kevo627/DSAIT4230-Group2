"""
Post-hoc gold-answer NLL scorer. For each row in pipeline.jsonl, rebuild the dialogue 
and measure how surprised the model is by the gold answer (one forward pass, no generation). 
Lower NLL = the clarification made the gold answer more expected. Prints a mean per strategy.

run: python scripts/run_nll_post_hoc.py --input results/pipeline.jsonl --load_in_4bit
"""

import argparse, json, os, sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import VLMWrapper
from main import build_final_answer_messages

ANSWER_REQUEST = "Now answer the original question. Reply with the short answer only."


def context_messages(row):
    """Pipeline's clarified dialogue, with a plain answer request as the last turn."""
    msgs = build_final_answer_messages(row)
    msgs[-1] = {"role": "user",
                "content": [{"type": "text",
                             "text": row["user_response"] + "\n\n" + ANSWER_REQUEST}]}
    return msgs


def gold_nll(model, messages, gold_answer):
    import torch
    from qwen_vl_utils import process_vision_info
    proc, mdl = model._processor, model._model

    text_ctx = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_full = text_ctx + gold_answer
    images, videos = process_vision_info(messages)

    enc = proc(text=[text_full], images=images, videos=videos, return_tensors="pt")
    ctx_ids = proc(text=[text_ctx], images=images, videos=videos,
                   return_tensors="pt").input_ids[0].tolist()
    full_ids = enc.input_ids[0].tolist()

    # boundary of the gold span (robust to BPE merges)
    n_ctx = 0
    for a, b in zip(ctx_ids, full_ids):
        if a != b:
            break
        n_ctx += 1
    if len(full_ids) - n_ctx <= 0:
        return None

    enc = {k: (v.to(mdl.device) if hasattr(v, "to") else v) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    labels[:, :n_ctx] = -100  # only the gold tokens are targets
    with torch.no_grad():
        return float(mdl(**enc, labels=labels).loss.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--load_in_4bit", action="store_true")
    args = p.parse_args()
    out_path = args.output or args.input.replace(".jsonl", "_nll.jsonl")

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "error" not in r:
                    rows.append(r)
    print(f"Loaded {len(rows)} rows")

    model = VLMWrapper(load_in_4bit=args.load_in_4bit)
    model.load()

    from tqdm import tqdm
    scored = []
    with open(out_path, "w") as out_f:
        for row in tqdm(rows, desc="NLL"):
            try:
                nll = gold_nll(model, context_messages(row), str(row["gold_answer"]))
            except Exception as e:
                nll = None
                print(f"skip id={row.get('id')}: {e}")
            rec = {"id": row.get("id"), "condition": row.get("condition"), "gold_nll": nll}
            scored.append(rec)
            out_f.write(json.dumps(rec) + "\n")

    print("\nMean gold NLL per strategy (lower = better):")
    by_cond = {}
    for r in scored:
        if r["gold_nll"] is not None:
            by_cond.setdefault(r["condition"], []).append(r["gold_nll"])
    for cond, vals in sorted(by_cond.items()):
        print(f"  {cond:>14}: {mean(vals):.4f}  (n={len(vals)})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()