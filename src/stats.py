"""
stats.py -- Paper 6.9 / Appendix B, C: significance tests and inter-annotator agreement.

CLI
    python -m src.stats human                      # Table 11, kappas, Wilcoxon from expert_evaluation/*.xlsx
    python -m src.stats bootstrap --a A.jsonl --b B.jsonl [--n 1000]   # paired bootstrap on char-BLEU

Library
    load_expert_scores, human_report, fleiss_kappa, paired_bootstrap_bleu, bootstrap_ci_bleu
"""
import argparse
import glob
import json
import os

import numpy as np

SYSTEMS = {"ALLaM-7B": "allam", "AraT5v2": "arat5", "AraBART": "arabart"}
# column indices in the experts' sheets: C=AraT5v2 score, D=ALLaM score, G=AraBART score
_COL = {"arat5": 2, "allam": 3, "arabart": 6}


# ---------------------------------------------------------------- human evaluation
def load_expert_scores(files):
    """Returns {system: int array (n_experts, 50)} and the list of source sentences."""
    import openpyxl
    scores = {k: [] for k in _COL}
    sources = None
    for f in files:
        rows = list(openpyxl.load_workbook(f, data_only=True).active.iter_rows(values_only=True))
        body = [r for r in rows[1:] if r[0] is not None and str(r[0]).strip()]
        sources = sources or [str(r[0]).strip() for r in body]
        for key, col in _COL.items():
            scores[key].append([int(float(str(r[col]).strip())) for r in body])
    return {k: np.array(v) for k, v in scores.items()}, sources


def counts_table(S):
    """Per expert: n(score 2), n(score 1), n(score 0), cumulative (paper Table 11)."""
    out = {}
    for name, key in SYSTEMS.items():
        rows = []
        for e in range(S[key].shape[0]):
            a = S[key][e]
            rows.append({"expert": e + 1, "score2": int((a == 2).sum()), "score1": int((a == 1).sum()),
                         "score0": int((a == 0).sum()), "cumulative": int(2 * (a == 2).sum() + (a == 1).sum())})
        out[name] = rows
    return out


def fleiss_kappa(mat, categories=(0, 1, 2)):
    """mat: raters x items. Returns (kappa, observed agreement Po, chance agreement Pe)."""
    n_r, n_i = mat.shape
    cnt = np.array([[(mat[:, i] == c).sum() for c in categories] for i in range(n_i)], float)
    p_i = ((cnt ** 2).sum(1) - n_r) / (n_r * (n_r - 1))
    p_bar, p_j = p_i.mean(), cnt.sum(0) / (n_r * n_i)
    p_e = (p_j ** 2).sum()
    return (p_bar - p_e) / (1 - p_e), p_bar, p_e


def pairwise_kappas(a, b):
    from sklearn.metrics import cohen_kappa_score
    return {"po": float((a == b).mean()), "kappa": float(cohen_kappa_score(a, b)),
            "kappa_w": float(cohen_kappa_score(a, b, weights="quadratic"))}


def cluster_bootstrap(values_by_sentence, stat, n=10000, seed=0):
    """values_by_sentence: (n_sentences, ...) array; resamples SENTENCES (ratings are nested in
    sentences). Returns (point, ci_low, ci_high) with a 95% percentile interval."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values_by_sentence)
    point = stat(v)
    draws = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def human_report(files, n_boot=10000, seed=0):
    from scipy.stats import wilcoxon
    S, _ = load_expert_scores(files)
    rep = {"counts": counts_table(S), "critical": {}, "wilcoxon": {}, "kappa": {}}
    for name, key in SYSTEMS.items():
        crit = (S[key] == 0)                                  # (experts, sentences)
        pooled = int(crit.sum())
        rate, lo, hi = cluster_bootstrap(crit.T.astype(float), lambda v: 100 * v.mean(), n=n_boot, seed=seed)
        rep["critical"][name] = {"count": pooled, "of": int(crit.size), "rate_pct": round(100 * pooled / crit.size, 1),
                                 "ci95_sentence_bootstrap": [round(lo, 1), round(hi, 1)]}
        rep["kappa"][name] = {"fleiss": round(fleiss_kappa(S[key])[0], 3)}
        for i, j in ((0, 1), (0, 2), (1, 2)):
            rep["kappa"][name][f"E{i+1}-E{j+1}"] = {k: round(v, 3) for k, v in pairwise_kappas(S[key][i], S[key][j]).items()}
    allr = np.concatenate([S[k] for k in S.keys()], axis=1)
    rep["kappa"]["global_fleiss"] = round(fleiss_kappa(allr)[0], 3)
    mean = {name: S[key].mean(0) for name, key in SYSTEMS.items()}
    for a, b in (("ALLaM-7B", "AraT5v2"), ("ALLaM-7B", "AraBART"), ("AraT5v2", "AraBART")):
        diff = mean[a] - mean[b]
        point, lo, hi = cluster_bootstrap(diff, np.mean, n=n_boot, seed=seed)
        p_auto = float(wilcoxon(mean[a], mean[b]).pvalue)                         # normal approx with ties (paper)
        p_exact = float(wilcoxon(mean[a], mean[b], method="exact").pvalue)
        rep["wilcoxon"][f"{a} vs {b}"] = {"mean_diff": round(point, 3), "ci95": [round(lo, 3), round(hi, 3)],
                                          "p_scipy_default": round(p_auto, 4), "p_exact": round(p_exact, 4)}
    cr = rep["critical"]
    rep["relative_critical_error_reduction_allam_vs_arat5_pct"] = round(
        100 * (cr["AraT5v2"]["count"] - cr["ALLaM-7B"]["count"]) / cr["AraT5v2"]["count"], 1)
    return rep


# ---------------------------------------------------------------- BLEU bootstrap
def _bleu_stats(hyps, refs):
    from sacrebleu.metrics import BLEU
    b = BLEU(tokenize="char")
    return b, np.array(b._extract_corpus_statistics(hyps, [refs]), dtype=float)


def _score(b, stats_sum):
    return b._compute_score_from_stats(list(stats_sum)).score


def paired_bootstrap_bleu(hyps_a, hyps_b, refs, n=1000, seed=0):
    """Paired bootstrap on corpus char-BLEU. Returns observed diff (A-B), 95% CI, % resamples A>B."""
    b, sa = _bleu_stats(hyps_a, refs)
    _, sb = _bleu_stats(hyps_b, refs)
    rng = np.random.default_rng(seed)
    N = len(refs)
    obs = _score(b, sa.sum(0)) - _score(b, sb.sum(0))
    diffs = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, N, N)
        diffs[k] = _score(b, sa[idx].sum(0)) - _score(b, sb[idx].sum(0))
    return {"obs_a": _score(b, sa.sum(0)), "obs_b": _score(b, sb.sum(0)), "diff": obs,
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "pct_a_better": float(100 * (diffs > 0).mean())}


def bootstrap_ci_bleu(hyps, refs, n=10000, seed=0):
    b, s = _bleu_stats(hyps, refs)
    rng = np.random.default_rng(seed)
    N = len(refs)
    draws = np.array([_score(b, s[rng.integers(0, N, N)].sum(0)) for _ in range(n)])
    return {"bleu": _score(b, s.sum(0)), "boot_mean": float(draws.mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


# ---------------------------------------------------------------- CLI
def _load_jsonl(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return [r["prediction"] for r in rows], [r["reference"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("human")
    h.add_argument("--files", nargs="*", default=sorted(glob.glob("expert_evaluation/expert*.xlsx")))   # expert1, expert2, expert3
    h.add_argument("--out", default="results/table11_human")
    h.add_argument("--n_boot", type=int, default=10000)
    b = sub.add_parser("bootstrap")
    b.add_argument("--a", required=True); b.add_argument("--b", required=True)
    b.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    if args.cmd == "human":
        rep = human_report(args.files, n_boot=args.n_boot)
        os.makedirs(args.out, exist_ok=True)
        json.dump(rep, open(os.path.join(args.out, "human_report.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        pa, refs = _load_jsonl(args.a); pb, _ = _load_jsonl(args.b)
        print(json.dumps(paired_bootstrap_bleu(pa, pb, refs, n=args.n), indent=2))


if __name__ == "__main__":
    main()
