"""
generate.py -- decoding helpers + run saving (paper 5.3 and 6.12).

* generate_seq2seq : batched beam search for AraT5v2 / AraBART (authors' settings, D8)
* generate_causal  : one sentence at a time for LLMs (D16), greedy or beam
* save_run         : writes predictions.jsonl, metrics.json, config copy, training log

torch/transformers are imported lazily so save_run works on a machine without them.
"""
import json
import os
import time

from src import metrics


def generate_seq2seq(model, tokenizer, sources, decoding, max_source_length=128,
                     batch_size=16, prefix=""):
    import torch
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(sources), batch_size):
        batch = [prefix + s.strip() for s in sources[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_source_length).to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=decoding.get("max_new_tokens", 128),
                num_beams=decoding.get("num_beams", 4),
                early_stopping=decoding.get("early_stopping", True),
                no_repeat_ngram_size=decoding.get("no_repeat_ngram_size", 3),
                length_penalty=decoding.get("length_penalty", 1.0),
            )
        outs += [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]
    return outs


def generate_causal(model, tokenizer, prompts, decoding, log_every=50):
    """One prompt at a time, no padding (D16): batching would put EOS tokens in
    input_ids and repetition_penalty would then penalise EOS."""
    import torch
    model.eval()
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outs, t0 = [], time.time()
    for k, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=decoding.get("max_new_tokens", 128),
                do_sample=decoding.get("do_sample", False),
                num_beams=decoding.get("num_beams", 1),
                repetition_penalty=decoding.get("repetition_penalty", 1.1),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen[0][enc["input_ids"].shape[1]:]
        outs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        if log_every and (k + 1) % log_every == 0:
            print(f"  generated {k + 1}/{len(prompts)}  ({time.time() - t0:.0f}s)", flush=True)
    return outs


def save_run(out_dir, test_items, preds, config=None, meta=None, log_history=None):
    """test_items: list of dicts with 'input','output'. preds: parallel list of strings."""
    os.makedirs(out_dir, exist_ok=True)
    assert len(test_items) == len(preds)
    refs = [d["output"] for d in test_items]
    inputs = [d["input"] for d in test_items]
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i, (d, p) in enumerate(zip(test_items, preds)):
            f.write(json.dumps({"id": i, "input": d["input"], "reference": d["output"],
                                "prediction": p}, ensure_ascii=False) + "\n")
    result = {"metrics": metrics.compute_all(preds, refs, inputs=inputs), "meta": meta or {}}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if config is not None:
        with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    if log_history is not None:
        with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as f:
            json.dump(log_history, f, ensure_ascii=False, indent=2)
    return result


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
