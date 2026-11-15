"""summarize.py -- one table of every finished run.
    python -m src.summarize                      # all result folders
    python -m src.summarize results/table7_seq2seq
"""
import glob
import json
import os
import sys


def main():
    roots = sys.argv[1:] or ["results/table6_zeroshot", "results/table7_seq2seq", "results/table8_lora"]
    print(f"{'run':<52}{'BLEU':>7}{'chrF++':>8}{'TER':>7}{'TokF1':>7}{'EM':>7}  notes")
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
            r = json.load(open(f, encoding="utf-8")); m, t = r["metrics"], r.get("meta", {})
            ck = os.path.basename(t["best_checkpoint"]) if t.get("best_checkpoint") else "-"
            note = f"ckpt={ck}" + (f" r={t['r']} trainable={t['trainable_pct']}%" if "r" in t else "")
            print(f"{t.get('run', os.path.basename(os.path.dirname(f))):<52}{m['bleu_char']:>7}{m['chrf_pp']:>8}"
                  f"{m['ter']:>7}{m['token_f1_set']:>7}{m['exact_match']:>7}  {note}")
    print(f"{'PAPER AraT5v2 (Table 7)':<52}{79.78:>7}{74.96:>8}{25.32:>7}{79.85:>7}{31.78:>7}")
    print(f"{'PAPER ALLaM-7B C2 (Table 8)':<52}{80.67:>7}{75.98:>8}{26.33:>7}{79.55:>7}{32.22:>7}")


if __name__ == "__main__":
    main()
