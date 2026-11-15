"""
data.py  --  Paper section 4: corpus loading, splitting, prompt formats.

Split design (see DECISIONS.md, D1):
  * The corpus is validated, shuffled with seed 42 and cut exactly as in the
    authors' scripts, so the TEST set (last 450 items) is the same 450 sentences.
  * The authors' scripts used 85/15 and took validation = test[:80]
    (validation leaked from test). The paper describes 80/5/15 = 2400/150/450
    with a separate validation set. We follow the paper:
        train = shuffled[:2400], val = shuffled[2400:2550], test = shuffled[2550:]

Usage:
    python -m src.data --raw data/raw/dataset_final_3000.json --out data/splits
"""
import argparse
import collections
import hashlib
import json
import os
import random

SEED = 42
N_TRAIN, N_VAL, N_TEST = 2400, 150, 450

# ---- Prompt formats ---------------------------------------------------------
# Training / fine-tuned inference (Alpaca-style, taken from the authors' code).
TRAIN_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
INFER_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

# AraT5v2 task prefix (authors' code; not mentioned in the paper).
ARAT5_PREFIX = "ترجم من اللهجة التونسية إلى العربية الفصحى: "


def format_train(item: dict) -> str:
    return TRAIN_TEMPLATE.format(
        instruction=item["instruction"].strip(),
        input=item["input"].strip(),
        output=item["output"].strip(),
    )


def format_infer(item: dict) -> str:
    return INFER_TEMPLATE.format(
        instruction=item["instruction"].strip(), input=item["input"].strip()
    )


# Zero-shot: each LLM gets its own native chat format (authors' code).
def zeroshot_prompt(model_key: str, item: dict) -> str:
    ins, inp = item["instruction"].strip(), item["input"].strip()
    if model_key == "allam7b":
        return format_infer(item)
    if model_key == "llama3_8b":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{ins}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if model_key == "mistral7b":
        return f"<s>[INST] {ins}\n\n{inp} [/INST]"
    raise ValueError(model_key)


# ---- Loading / splitting ----------------------------------------------------
def is_valid(item) -> bool:
    return (
        isinstance(item, dict)
        and item.get("instruction", "").strip()
        and item.get("input", "").strip()
        and item.get("output", "").strip()
    )


def load_raw(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", list(data.values()))
    return [d for d in data if is_valid(d)]


def canonical_instruction(raw: list) -> str:
    """The corpus has 53 instructions with mixed-script typos; use the majority one."""
    return collections.Counter(d["instruction"].strip() for d in raw).most_common(1)[0][0]


def make_splits(raw: list, fix_instructions: bool = True):
    raw = [dict(d) for d in raw]
    if fix_instructions:
        canon = canonical_instruction(raw)
        for d in raw:
            d["instruction"] = canon
    random.seed(SEED)           # same procedure as the authors' scripts
    random.shuffle(raw)
    assert len(raw) == N_TRAIN + N_VAL + N_TEST, f"expected 3000 pairs, got {len(raw)}"
    train = raw[:N_TRAIN]
    val = raw[N_TRAIN:N_TRAIN + N_VAL]
    test = raw[N_TRAIN + N_VAL:]
    return train, val, test


def make_splits_authors(raw: list, fix_instructions: bool = True):
    """The authors' ACTUAL protocol (from their scripts): same shuffle, 85/15 = 2550/450,
    validation = test[:80] (a slice of the test set!). Only for the ablation that measures
    what this leakage and the 150 extra training sentences are worth."""
    raw = [dict(d) for d in raw]
    if fix_instructions:
        canon = canonical_instruction(raw)
        for d in raw:
            d["instruction"] = canon
    random.seed(SEED)
    random.shuffle(raw)
    train, test = raw[:2550], raw[2550:]
    return train, test[:80], test


def load_split(split_dir: str, name: str) -> list:
    with open(os.path.join(split_dir, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


# ---- Audit ------------------------------------------------------------------
def audit(train, val, test) -> dict:
    strip = lambda s: s.strip()
    tr_out = {strip(d["output"]) for d in train + val}
    tr_in = {strip(d["input"]) for d in train + val}
    allp = train + val + test
    return {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "identity_pairs_all": sum(strip(d["input"]) == strip(d["output"]) for d in allp),
        "identity_pairs_test": sum(strip(d["input"]) == strip(d["output"]) for d in test),
        "duplicate_outputs_all": len(allp) - len({strip(d["output"]) for d in allp}),
        "test_inputs_in_train_or_val": sum(strip(d["input"]) in tr_in for d in test),
        "test_outputs_in_train_or_val": sum(strip(d["output"]) in tr_out for d in test),
    }


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/dataset_final_3000.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol", choices=["paper", "authors"], default="paper")
    args = ap.parse_args()
    args.out = args.out or ("data/splits" if args.protocol == "paper" else "data/splits_authors")

    raw = load_raw(args.raw)
    train, val, test = make_splits(raw) if args.protocol == "paper" else make_splits_authors(raw)
    os.makedirs(args.out, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)
    manifest = {"protocol": args.protocol, "seed": SEED, "raw_sha256": sha256(args.raw), "audit": audit(train, val, test)}
    with open(os.path.join(args.out, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
