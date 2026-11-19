"""
plot_curves.py -- training curves for every run (replaces the authors' ten plot scripts; paper Table 9 / Figs 1-3).

    python -m src.plot_curves                      # all runs that have a training_log.json
    python -m src.plot_curves results/table8_lora/lora_allam7b_c2_seed42

Per run: <run>/curves.png (train vs validation loss | validation BLEU if logged | both together),
plus results/table9_training_dynamics/table9.md (validation loss / BLEU at every evaluation step).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def history(log):
    """-> list of dicts(step, epoch, train_loss, val_loss, val_bleu) at every evaluation."""
    rows, last_train = [], None
    for e in log:
        if "loss" in e and "eval_loss" not in e:
            last_train = e["loss"]
        if "eval_loss" in e:
            rows.append({"step": e["step"], "epoch": e.get("epoch"), "train_loss": last_train,
                         "val_loss": e["eval_loss"], "val_bleu": e.get("eval_bleu")})
    return rows


def best_step(run_dir):
    try:
        m = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))["meta"]
        ck = m.get("best_checkpoint")
        return int(os.path.basename(ck).split("-")[-1]) if ck else None
    except Exception:
        return None


def plot_run(run_dir):
    p = os.path.join(run_dir, "training_log.json")
    if not os.path.exists(p):
        return None
    rows = history(json.load(open(p, encoding="utf-8")))
    if not rows:
        return None
    steps = [r["step"] for r in rows]
    has_bleu = any(r["val_bleu"] is not None for r in rows)
    bs = best_step(run_dir)
    ncol = 3 if has_bleu else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
    axes = [axes] if ncol == 1 else list(axes)
    ax = axes[0]
    ax.plot(steps, [r["train_loss"] for r in rows], "o-", label="train loss (last logged)")
    ax.plot(steps, [r["val_loss"] for r in rows], "s--", label="validation loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Loss")
    if bs:
        ax.axvline(bs, color="purple", ls=":", label=f"kept checkpoint (step {bs})")
    ax.legend(fontsize=8)
    if has_bleu:
        v = [(r["step"], r["val_bleu"]) for r in rows if r["val_bleu"] is not None]
        axes[1].plot(*zip(*v), "D-", color="green")
        axes[1].set_xlabel("step"); axes[1].set_ylabel("validation BLEU (char)"); axes[1].set_title("Validation BLEU")
        if bs:
            axes[1].axvline(bs, color="purple", ls=":")
        ax2 = axes[2]
        ax2.plot(steps, [r["val_loss"] for r in rows], "s--", color="tab:orange")
        ax2.set_xlabel("step"); ax2.set_ylabel("validation loss", color="tab:orange")
        tw = ax2.twinx(); tw.plot(*zip(*v), "D-", color="green"); tw.set_ylabel("validation BLEU", color="green")
        ax2.set_title("Loss vs BLEU")
    fig.suptitle(os.path.basename(run_dir.rstrip("/")), fontsize=10)
    fig.tight_layout()
    out = os.path.join(run_dir, "curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, rows


def main():
    dirs = sys.argv[1:] or sorted({os.path.dirname(p) for p in glob.glob("results/*/*/training_log.json")
                                   if "_dryrun" not in p})
    table = ["| run | step | epoch | train loss | val loss | val BLEU |", "|---|---|---|---|---|---|"]
    for d in dirs:
        res = plot_run(d)
        if not res:
            continue
        out, rows = res
        print("wrote", out)
        for r in rows:
            f = lambda v, nd=3: "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
            table.append(f"| {os.path.basename(d)} | {r['step']} | {f(r['epoch'], 2)} | {f(r['train_loss'])} | "
                         f"{f(r['val_loss'], 4)} | {f(r['val_bleu'], 2)} |")
    os.makedirs("results/table9_training_dynamics", exist_ok=True)
    open("results/table9_training_dynamics/table9.md", "w", encoding="utf-8").write("\n".join(table) + "\n")
    print("wrote results/table9_training_dynamics/table9.md")


if __name__ == "__main__":
    main()
