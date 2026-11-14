"""
Sanity checks. Run from the repo root:  python -m tests.test_against_stored

1. Our split's TEST set equals the 450 test items stored in the authors' result
   files (one known reference was corrected in the corpus afterwards).
2. src/metrics.py recomputes the authors' stored metrics from their stored
   predictions (AraT5v2, AraBART, ALLaM Config 3).
"""
import json
import os
import sys

from src import data, metrics

ORIG = "original_results"
FILES = {
    "AraT5v2": "arat5v2_evaluation_results.json",
    "AraBART": "arabart_evaluation_results.json",
    "ALLaM-C3": "evaluation_results_allam_5epochs.json",
}
ok = True


def check(label, mine, stored, tol):
    global ok
    good = abs(mine - stored) <= tol
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} {label:<14} mine={mine:<8} stored={stored:<8}")


raw = data.load_raw("data/raw/dataset_final_3000.json")
train, val, test = data.make_splits(raw)
print(f"split sizes: train={len(train)} val={len(val)} test={len(test)}")
assert (len(train), len(val), len(test)) == (2400, 150, 450)

for name, fn in FILES.items():
    r = json.load(open(os.path.join(ORIG, fn), encoding="utf-8"))
    P = r["predictions"]
    same = sum(a["input_tun"].strip() == b["input"].strip() for a, b in zip(P, test))
    print(f"\n{name}: test inputs matching our split = {same}/450")
    ok &= same == 450
    m = metrics.compute_all([x["predicted"] for x in P], [x["reference"] for x in P],
                            inputs=[x["input_tun"] for x in P])
    check("BLEU(char)", m["bleu_char"], r["corpus_bleu"], 0.01)
    check("chrF++", m["chrf_pp"], r["corpus_chrf_pp"], 0.01)
    check("TER", m["ter"], r["corpus_ter"], 0.01)
    check("TokenF1(set)", m["token_f1_set"], round(100 * r["avg_token_f1"], 2), 0.02)
    check("ExactMatch", m["exact_match"], round(100 * r["exact_match_acc"], 2), 0.02)
    print(f"  extra: BLEU(13a)={m['bleu_13a']}  TokenF1(multiset)={m['token_f1_multiset']}  "
          f"EM excl. identity pairs={m['exact_match_excl_identity']}")

b = metrics.copy_baseline([d["input"] for d in test], [d["output"] for d in test])
print("\nCopy-input baseline (no model):", {k: b[k] for k in ("bleu_char", "bleu_13a", "chrf_pp", "ter", "exact_match")})
print("Audit:", data.audit(train, val, test))
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
