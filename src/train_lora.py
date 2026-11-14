"""
train_lora.py -- Paper 5.3 / Table 5: QLoRA fine-tuning of ALLaM-7B, LLaMA 3-8B, Mistral-7B.

    python -m src.train_lora --config configs/lora_allam7b_c2.yaml --dry_run   # ~3-5 min smoke test
    python -m src.train_lora --config configs/lora_allam7b_c2.yaml             # real run
    python -m src.train_lora --config configs/lora_allam7b_c2.yaml --es_impl authors_proxy

Early-stopping metric (config early_stopping.metric_impl, D12/D19):
    generation     (default) greedy-decode the 150 VALIDATION sentences, char-BLEU vs references.
    authors_proxy  the authors' compute_metrics: argmax of teacher-forced logits over the WHOLE
                   sequence (prompt+response), unshifted. Kept only to measure its effect.

Outputs: results/table8_lora/<run>/{predictions.jsonl,metrics.json,config_used.json,training_log.json}
Adapter: checkpoints/<run>/best_adapter (git-ignored).
Needs HF_TOKEN in the environment for gated models (LLaMA 3).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # D17

import argparse
import gc
import inspect
import json
import time

import yaml
from sacrebleu.metrics import BLEU

from src import data
from src.env import check_env
from src.generate import generate_causal, save_run


# ---- data preparation (pure python: unit-testable without torch) -------------
def tokenize_items(tokenizer, items, max_len):
    """Alpaca text + explicit EOS id appended (D18). Default special tokens (BOS) are
    added, exactly as at inference time."""
    rows, truncated = [], 0
    for d in items:
        ids = tokenizer(data.format_train(d), truncation=True,
                        max_length=max_len - 1)["input_ids"]
        if len(ids) >= max_len - 1:
            truncated += 1
        rows.append({"input_ids": ids + [tokenizer.eos_token_id]})
    if truncated:
        print(f"WARNING: {truncated} examples hit max_seq_length={max_len}", flush=True)
    return rows


def pad_batch(feats, pad_id):
    """Right-pad. Labels = input ids with padding -100 (the real EOS stays supervised even
    when pad_id == eos_id: DataCollatorForLanguageModeling would mask it)."""
    L = max(len(f["input_ids"]) for f in feats)
    ids, att, lab = [], [], []
    for f in feats:
        x, n = f["input_ids"], L - len(f["input_ids"])
        ids.append(x + [pad_id] * n)
        att.append([1] * len(x) + [0] * n)
        lab.append(x + [-100] * n)
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        return {k: torch.tensor(v) for k, v in pad_batch(feats, self.pad_id).items()}


# ---- validation BLEU ---------------------------------------------------------
def make_val_bleu_fn(tokenizer, val_items, max_new_tokens=128, batch_size=16):
    """Generation-based char-BLEU on the validation sentences (greedy, left-padded batches,
    repetition_penalty=1.0 for speed/EOS safety; test decoding uses the paper settings)."""
    import torch
    prompts = [data.format_infer(d) for d in val_items]
    refs = [d["output"].strip() for d in val_items]
    bleu = BLEU(tokenize="char")

    def fn(model):
        was_training = model.training
        model.eval()
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        preds = []
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                     max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=1.0, use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
            n = enc["input_ids"].shape[1]
            preds += [tokenizer.decode(o[n:], skip_special_tokens=True).strip() for o in out]
        tokenizer.padding_side = old_side
        if was_training:
            model.train()
        return round(bleu.corpus_score(preds, [refs]).score, 2)
    return fn


def make_proxy_metrics(tokenizer):
    """The authors' compute_metrics, reproduced faithfully (including its flaws)."""
    import numpy as np
    bleu = BLEU(tokenize="char")

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(-1)

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        pred_ids = np.where(pred_ids < 0, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        dp = [p.strip() for p in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        dl = [l.strip() for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        return {"bleu": round(bleu.corpus_score(dp, [dl]).score, 2)}
    return preprocess_logits, compute_metrics


def build_trainer_class():
    from transformers import Trainer

    class BleuTrainer(Trainer):
        """Adds eval_bleu to the metrics dict BEFORE logging/early stopping/best-model logic."""
        def __init__(self, *a, bleu_fn=None, **k):
            super().__init__(*a, **k)
            self.bleu_fn = bleu_fn

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                            ignore_keys=None, metric_key_prefix="eval"):
            out = super().evaluation_loop(dataloader, description,
                                          prediction_loss_only=prediction_loss_only,
                                          ignore_keys=ignore_keys,
                                          metric_key_prefix=metric_key_prefix)
            if self.bleu_fn is not None:
                t = time.time()
                out.metrics[f"{metric_key_prefix}_bleu"] = self.bleu_fn(self.model)
                print(f"  val BLEU (generation) = {out.metrics[f'{metric_key_prefix}_bleu']} "
                      f"[{time.time() - t:.0f}s]", flush=True)
            return out
    return BleuTrainer


# ---- main run ----------------------------------------------------------------
def run(cfg, seed, args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              EarlyStoppingCallback, TrainingArguments, set_seed)

    check_env()
    tr, lo, q, es = cfg["training"], cfg["lora"], cfg["quantization"], cfg["early_stopping"]
    impl = args.es_impl or es.get("metric_impl", "generation")
    name = cfg["name"] + (f"_{impl}" if es["enabled"] and impl != "generation" else "") \
        + f"_seed{seed}" + ("_dryrun" if args.dry_run else "")
    out_dir = os.path.join(args.out_root if not args.dry_run else "results/_dryrun", name)
    ckpt_dir = os.path.join(args.ckpt_root, name)
    print(f"\n{'=' * 70}\nRUN {name}\n{'=' * 70}", flush=True)

    set_seed(seed)
    train = data.load_split(cfg["splits_dir"], "train")
    val = data.load_split(cfg["splits_dir"], "val")
    test = data.load_split(cfg["splits_dir"], "test")
    epochs, eval_steps, save_steps = tr["max_epochs"], tr["eval_steps"], tr["save_steps"]
    if args.dry_run:
        train, val, test, epochs, eval_steps, save_steps = train[:64], val[:32], test[:8], 1, 2, 2

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], token=token,
                                        trust_remote_code=cfg.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cdtype = getattr(torch, q["compute_dtype"])
    bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=cdtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], quantization_config=bnb, device_map={"": 0}, torch_dtype=cdtype,
        token=token, trust_remote_code=cfg.get("trust_remote_code", False))
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
        target_modules=lo["target_modules"], bias=lo["bias"], task_type="CAUSAL_LM"))
    try:
        n_train_p, n_all_p = model.get_nb_trainable_parameters()
    except Exception:
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all_p = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train_p:,} / {n_all_p:,} = {100 * n_train_p / n_all_p:.3f}%", flush=True)

    train_rows = tokenize_items(tok, train, tr["max_seq_length"])
    val_rows = tokenize_items(tok, val, tr["max_seq_length"])

    use_bleu = es["enabled"]
    metric = "eval_bleu" if use_bleu else "eval_loss"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strat_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    ta = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=tr["per_device_batch_size"], per_device_eval_batch_size=2,
        gradient_accumulation_steps=tr["grad_accum_steps"], num_train_epochs=epochs,
        learning_rate=float(tr["learning_rate"]), lr_scheduler_type=tr["lr_scheduler"],
        warmup_steps=tr["warmup_steps"], weight_decay=tr["weight_decay"], optim=tr["optimizer"],
        fp16=(q["compute_dtype"] == "float16"), bf16=(q["compute_dtype"] == "bfloat16"),
        gradient_checkpointing=tr["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, eval_steps=eval_steps, save_steps=save_steps, save_strategy="steps",
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model=metric,
        greater_is_better=use_bleu, remove_unused_columns=False, dataloader_num_workers=0,
        report_to="none", seed=seed, **{strat_key: "steps"})

    tk = ("processing_class" if "processing_class" in
          inspect.signature(build_trainer_class().__init__).parameters else "tokenizer")
    extra = {}
    if use_bleu and impl == "generation":
        extra["bleu_fn"] = make_val_bleu_fn(tok, val, max_new_tokens=cfg["decoding"]["max_new_tokens"])
    elif use_bleu and impl == "authors_proxy":
        extra["preprocess_logits_for_metrics"], extra["compute_metrics"] = make_proxy_metrics(tok)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=es["patience"])] if use_bleu else []
    trainer = build_trainer_class()(
        model=model, args=ta, train_dataset=train_rows, eval_dataset=val_rows,
        data_collator=Collator(tok.pad_token_id), callbacks=callbacks, **extra, **{tk: tok})
    print(f"train={len(train)} val={len(val)} test={len(test)} | effective batch="
          f"{tr['per_device_batch_size'] * tr['grad_accum_steps']} | r={lo['r']} alpha={lo['alpha']} "
          f"dropout={lo['dropout']} wd={tr['weight_decay']} | epochs<={epochs} | "
          f"selection={metric} ({impl if use_bleu else 'no early stopping'})", flush=True)

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 1)

    adapter_dir = os.path.join(ckpt_dir, "best_adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    m = trainer.model
    m.config.use_cache = True
    m.eval()
    t1 = time.time()
    preds = generate_causal(m, tok, [data.format_infer(d) for d in test], cfg["decoding"])
    log = trainer.state.log_history
    meta = {
        "run": name, "model_id": cfg["model_id"], "seed": seed, "r": lo["r"], "alpha": lo["alpha"],
        "dropout": lo["dropout"], "weight_decay": tr["weight_decay"],
        "trainable_params": n_train_p, "total_params": n_all_p,
        "trainable_pct": round(100 * n_train_p / n_all_p, 3),
        "selection_metric": metric, "selection_impl": impl if use_bleu else None,
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric,
        "epochs_run": trainer.state.epoch, "global_step": trainer.state.global_step,
        "stopped_early": trainer.state.global_step < trainer.state.max_steps,
        "val_bleu_history": [(e["step"], e["eval_bleu"]) for e in log if "eval_bleu" in e],
        "val_loss_history": [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e],
        "train_seconds": round(train_secs), "generate_seconds": round(time.time() - t1),
        "peak_gpu_gb": peak_gb, "n_test": len(test), "dry_run": args.dry_run,
        "decoding": cfg["decoding"],
    }
    result = save_run(out_dir, test, preds, config=cfg, meta=meta, log_history=log)
    print("\nTEST METRICS:", json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out_dir}\nadapter -> {adapter_dir}", flush=True)
    del trainer, model, m
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="64 train / 32 val / 8 test, 1 epoch")
    ap.add_argument("--es_impl", choices=["generation", "authors_proxy"], default=None)
    ap.add_argument("--out_root", default="results/table8_lora")
    ap.add_argument("--ckpt_root", default="checkpoints")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg, args.seed if args.seed is not None else cfg.get("seed", 42), args)


if __name__ == "__main__":
    main()
