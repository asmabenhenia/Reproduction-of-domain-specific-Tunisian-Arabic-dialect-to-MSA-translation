"""Run: python -m tests.test_stats_against_paper
Checks src/stats.py against numbers the paper prints (Sec. 6.9, Appendix B/C)."""
import glob
import json

import numpy as np
from sacrebleu.metrics import BLEU

from src import stats

ok = True
def check(label, mine, paper, tol):
    global ok
    good = abs(mine - paper) <= tol
    ok &= good
    print(f"  {'OK  ' if good else 'FAIL'} {label:<44} mine={mine:<9.4g} paper={paper:<9.4g} tol={tol}")

files = sorted(glob.glob("expert_evaluation/expert*.xlsx"))   # expert1, expert2, expert3 in order
assert len(files) == 3, files
rep = stats.human_report(files, n_boot=2000)

print("Pooled critical errors (paper 5, 14, 30 of 150):")
for n, c in (("ALLaM-7B", 5), ("AraT5v2", 14), ("AraBART", 30)):
    check(n, rep["critical"][n]["count"], c, 0)
print("Fleiss kappa (paper 0.827, 0.817, 0.772; global 0.806):")
check("ALLaM-7B", rep["kappa"]["ALLaM-7B"]["fleiss"], 0.827, 0.001)
check("AraT5v2", rep["kappa"]["AraT5v2"]["fleiss"], 0.817, 0.001)
check("AraBART", rep["kappa"]["AraBART"]["fleiss"], 0.772, 0.001)
check("global", rep["kappa"]["global_fleiss"], 0.806, 0.001)
print("Wilcoxon on per-sentence mean scores (paper p=0.057 / 0.0013 / 0.043; diff .187 / .433 / .247):")
for k, p, d in (("ALLaM-7B vs AraT5v2", 0.057, 0.187), ("ALLaM-7B vs AraBART", 0.0013, 0.433), ("AraT5v2 vs AraBART", 0.043, 0.247)):
    check("p " + k, rep["wilcoxon"][k]["p_scipy_default"], p, 0.0006)
    check("mean diff " + k, rep["wilcoxon"][k]["mean_diff"], d, 0.001)
check("relative critical-error reduction (%)", rep["relative_critical_error_reduction_allam_vs_arat5_pct"], 64.3, 0.1)

# bootstrap on the authors' stored predictions
A = json.load(open("original_results/arat5v2_evaluation_results.json", encoding="utf-8"))["predictions"]
B = json.load(open("original_results/arabart_evaluation_results.json", encoding="utf-8"))["predictions"]
C = json.load(open("original_results/evaluation_results_allam_5epochs.json", encoding="utf-8"))["predictions"]
refs = [x["reference"] for x in A]
b, s = stats._bleu_stats([x["predicted"] for x in A], refs)
check("fast BLEU == sacrebleu corpus BLEU (AraT5v2)", stats._score(b, s.sum(0)), BLEU(tokenize="char").corpus_score([x["predicted"] for x in A], [refs]).score, 1e-6)
print("Paired bootstrap AraT5v2 vs AraBART (paper +17.29, CI [15.70, 18.77], 100%):")
r = stats.paired_bootstrap_bleu([x["predicted"] for x in A], [x["predicted"] for x in B], refs, n=1000)
check("diff", r["diff"], 17.29, 0.02); check("CI low", r["ci95"][0], 15.70, 0.4); check("CI high", r["ci95"][1], 18.77, 0.4); check("% better", r["pct_a_better"], 100.0, 0.1)
print("Appendix C, ALLaM r=16 5 epochs (paper BLEU 79.88, mean 79.91, CI [78.17, 81.49]):")
r = stats.bootstrap_ci_bleu([x["predicted"] for x in C], [x["reference"] for x in C], n=3000)
check("BLEU", r["bleu"], 79.88, 0.01); check("CI low", r["ci95"][0], 78.17, 0.4); check("CI high", r["ci95"][1], 81.49, 0.4)
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
