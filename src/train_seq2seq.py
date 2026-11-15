"""
train_seq2seq.py -- Paper 5.2 / Table 4: full fine-tuning of AraT5v2 and AraBART.

    python -m src.train_seq2seq --config configs/seq2seq_arat5v2.yaml
    python -m src.train_seq2seq --config configs/seq2seq_arabart.yaml --sweep      # 3 learning rates
    python -m src.train_seq2seq --config configs/seq2seq_arat5v2.yaml --dry_run    # 1-minute smoke test

Outputs (per run) in results/table7_seq2seq/<run>/ :
    predictions.jsonl, metrics.json, config_used.json, training_log.json
Model checkpoints go to checkpoints/<run>/ (git-ignored).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # D17: keep effective batch = 32

import argparse
import gc
import inspect
import json
import time

import yaml

from src import data
from src.env import check_env
from src.generate import generate_seq2seq, save_run


def build_datasets(tokenizer, train, val, prefix, max_src, max_tgt):
    from datasets import Dataset

    def preprocess(batch):
        sources = [prefix + s.strip() for s in batch["input"]]
        targets = [t.strip() for t in batch["output"]]
        enc = tokenizer(sources, max_length=max_src, truncation=True, padding=False)
        lab = tokenizer(text_target=targets, max_length=max_tgt, truncation=True, padding=False)
        enc["labels"] = lab["input_ids"]
        return enc

    def make(items):
        ds = Dataset.from_list([{"input": d["input"], "output": d["output"]} for d in items])
        return ds.map(preprocess, batched=True, batch_size=256, remove_columns=ds.column_names)

    return make(train), make(val)


def run_one(cfg, lr, seed, args):
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              set_seed)

    check_env()
    tr = cfg["training"]
    name = (f"{cfg['name']}_lr{lr:g}_seed{seed}" + (f"_{args.tag}" if args.tag else "")
            + ("_dryrun" if args.dry_run else ""))
    splits_dir = args.splits_dir or cfg["splits_dir"]
    save_steps = args.save_steps or tr["save_steps"]
    out_dir = os.path.join(args.out_root if not args.dry_run else "results/_dryrun", name)
    ckpt_dir = os.path.join(args.ckpt_root, name)
    print(f"\n{'=' * 70}\nRUN {name}\n{'=' * 70}", flush=True)

    set_seed(seed)
    train = data.load_split(splits_dir, "train")
    val = data.load_split(splits_dir, "val")
    test = data.load_split(splits_dir, "test")
    epochs = tr["max_epochs"]
    if args.dry_run:
        train, val, test, epochs = train[:64], val[:32], test[:16], 1

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_id"])   # fp32 weights (D9)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"| visible GPUs: {torch.cuda.device_count()}", flush=True)

    prefix = cfg.get("source_prefix") or ""
    train_ds, val_ds = build_datasets(tokenizer, train, val, prefix,
                                      tr["max_source_length"], tr["max_target_length"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100,
                                      pad_to_multiple_of=8)

    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strat_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    prec = tr.get("precision", "fp32")
    ta = Seq2SeqTrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=tr["per_device_batch_size"],
        gradient_accumulation_steps=tr["grad_accum_steps"],
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type=tr["lr_scheduler"],
        warmup_steps=tr["warmup_steps"],
        weight_decay=tr["weight_decay"],
        fp16=(prec == "fp16"),
        bf16=(prec == "bf16"),
        predict_with_generate=False,          # selection on val loss (D12); faster
        logging_steps=20,
        eval_steps=2 if args.dry_run else tr["eval_steps"],     # dry run: exercise eval/save
        save_steps=2 if args.dry_run else save_steps,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
        **{strat_key: "steps"},
    )
    tk = ("processing_class"
          if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
          else "tokenizer")
    trainer = Seq2SeqTrainer(
        model=model, args=ta, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tr["early_stopping_patience"])],
        **{tk: tokenizer},
    )
    eff = tr["per_device_batch_size"] * tr["grad_accum_steps"] * max(1, ta.n_gpu)
    print(f"train={len(train)} val={len(val)} test={len(test)} | effective batch={eff} "
          f"| lr={lr:g} | epochs<={epochs}", flush=True)

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    dec = cfg["decoding"]
    t1 = time.time()
    preds = generate_seq2seq(trainer.model, tokenizer, [d["input"] for d in test], dec,
                             max_source_length=tr["max_source_length"], prefix=prefix)
    meta = {
        "run": name, "model_id": cfg["model_id"], "lr": lr, "seed": seed,
        "effective_batch": eff, "epochs_run": trainer.state.epoch,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_loss": trainer.state.best_metric,
        "train_seconds": round(train_secs), "generate_seconds": round(time.time() - t1),
        "n_test": len(test), "dry_run": args.dry_run, "decoding": dec,
        "splits_dir": splits_dir, "save_steps": save_steps, "n_train": len(train), "n_val": len(val),
    }
    result = save_run(out_dir, test, preds, config=cfg, meta=meta,
                      log_history=trainer.state.log_history)
    print("\nTEST METRICS:", json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr", type=float, default=None, help="override learning rate")
    ap.add_argument("--sweep", action="store_true", help="run every lr in training.lr_sweep")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="64 train / 32 val / 16 test, 1 epoch")
    ap.add_argument("--splits_dir", default=None, help="e.g. data/splits_authors (ablation)")
    ap.add_argument("--save_steps", type=int, default=None, help="e.g. 100 = save at every eval")
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. save100")
    ap.add_argument("--out_root", default="results/table7_seq2seq")
    ap.add_argument("--ckpt_root", default="checkpoints")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    if args.sweep:
        lrs = [float(x) for x in cfg["training"]["lr_sweep"]]
    else:
        lrs = [args.lr if args.lr is not None else float(cfg["training"]["learning_rate"])]
    for lr in lrs:
        run_one(cfg, lr, seed, args)


if __name__ == "__main__":
    main()
