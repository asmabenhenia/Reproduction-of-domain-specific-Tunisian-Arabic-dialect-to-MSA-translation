"""
neural_eval.py -- Paper 6.11 / Table 12: COMET, COMETKiwi-DA, BLEURT-20 on saved predictions.

    python -m src.neural_eval --all                       # every run folder under results/
    python -m src.neural_eval results/table7_seq2seq/seq2seq_arat5v2_lr0.0005_seed42 [--skip_kiwi]

Writes neural_metrics.json inside each run folder. Needs a GPU and internet. Run it in its OWN Kaggle
session (unbabel-comet pins library versions). COMETKiwi needs HF_TOKEN + the licence accepted at
huggingface.co/Unbabel/wmt22-cometkiwi-da.
"""
import argparse
import glob
import json
import os

from src import metrics


def evaluate_dir(d, skip_kiwi=False, skip_bleurt=False, skip_comet=False):
    rows = [json.loads(l) for l in open(os.path.join(d, "predictions.jsonl"), encoding="utf-8")]
    src = [r["input"] for r in rows]; mt = [r["prediction"] for r in rows]; ref = [r["reference"] for r in rows]
    out = {"n": len(rows)}
    for key, skip, fn in (
        ("comet", skip_comet, lambda: metrics.comet_scores(src, mt, ref)[0]),
        ("cometkiwi_da", skip_kiwi, lambda: metrics.cometkiwi_scores(src, mt)[0]),
        ("bleurt", skip_bleurt, lambda: metrics.bleurt_scores(mt, ref)[0]),
    ):
        if skip:
            continue
        try:
            out[key] = round(float(fn()), 4)
        except Exception as e:                       # one failing metric must not stop the others
            out[key] = None
            out[key + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    json.dump(out, open(os.path.join(d, "neural_metrics.json"), "w", encoding="utf-8"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip_kiwi", action="store_true")
    ap.add_argument("--skip_bleurt", action="store_true")
    ap.add_argument("--skip_comet", action="store_true")
    args = ap.parse_args()
    dirs = list(args.dirs)
    if args.all:
        dirs += [os.path.dirname(p) for p in sorted(glob.glob("results/*/*/predictions.jsonl"))
                 if "_dryrun" not in p]
    assert dirs, "nothing to evaluate"
    print(f"{'run':<58}{'COMET':>8}{'Kiwi':>8}{'BLEURT':>8}")
    for d in dirs:
        o = evaluate_dir(d, args.skip_kiwi, args.skip_bleurt, args.skip_comet)
        f = lambda v: f"{v:>8}" if v is not None else f"{'ERR':>8}"
        print(f"{os.path.basename(d):<58}{f(o.get('comet'))}{f(o.get('cometkiwi_da'))}{f(o.get('bleurt'))}")
        for k, v in o.items():
            if k.endswith("_error"):
                print("   ", k, v)


if __name__ == "__main__":
    main()
