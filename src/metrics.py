"""
metrics.py  --  Paper sections 5.4 and 6.11.

Lexical metrics (sacrebleu):
    BLEU  corpus-level, tokenize="char"   (paper's headline BLEU)
    BLEU  corpus-level, default 13a       (extra: NOT in the paper, reported for transparency)
    chrF++ word_order=2,  TER,  Token F1,  Exact Match

Token F1 / Exact Match follow the authors' code (DECISIONS.md D6):
    * Token F1 (set)      : set-based overlap on whitespace tokens, mean over sentences
    * Token F1 (multiset) : extra, counts repeated tokens
    * Exact Match         : equality after stripping trailing "،" and "."

Neural metrics (COMET, COMETKiwi-DA, BLEURT-20) need GPU + model downloads;
they are imported lazily so the lexical metrics work anywhere.
"""
from collections import Counter

from sacrebleu.metrics import BLEU, CHRF, TER


# ---- sentence-level helpers -------------------------------------------------
def exact_match(p: str, r: str) -> bool:
    return p.strip().rstrip("،.") == r.strip().rstrip("،.")


def token_f1_set(p: str, r: str) -> float:
    pt, rt = set(p.strip().split()), set(r.strip().split())
    if not pt or not rt:
        return float(pt == rt)
    common = pt & rt
    if not common:
        return 0.0
    prec, rec = len(common) / len(pt), len(common) / len(rt)
    return 2 * prec * rec / (prec + rec)


def token_f1_multiset(p: str, r: str) -> float:
    pc, rc = Counter(p.strip().split()), Counter(r.strip().split())
    if not pc or not rc:
        return float(pc == rc)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(pc.values()), overlap / sum(rc.values())
    return 2 * prec * rec / (prec + rec)


# ---- corpus-level -----------------------------------------------------------
def compute_all(preds, refs, inputs=None) -> dict:
    """preds/refs: parallel lists of strings. inputs: optional dialect sources
    (used to report Exact Match with and without identity pairs)."""
    assert len(preds) == len(refs) and len(preds) > 0
    n = len(preds)
    em = [exact_match(p, r) for p, r in zip(preds, refs)]
    out = {
        "n": n,
        "bleu_char": round(BLEU(tokenize="char").corpus_score(preds, [refs]).score, 2),
        "bleu_13a": round(BLEU().corpus_score(preds, [refs]).score, 2),
        "chrf_pp": round(CHRF(word_order=2).corpus_score(preds, [refs]).score, 2),
        "ter": round(TER().corpus_score(preds, [refs]).score, 2),
        "token_f1_set": round(100 * sum(token_f1_set(p, r) for p, r in zip(preds, refs)) / n, 2),
        "token_f1_multiset": round(100 * sum(token_f1_multiset(p, r) for p, r in zip(preds, refs)) / n, 2),
        "exact_match": round(100 * sum(em) / n, 2),
    }
    if inputs is not None:
        ident = [i.strip() == r.strip() for i, r in zip(inputs, refs)]
        keep = [k for k, idn in enumerate(ident) if not idn]
        out["n_identity_pairs"] = sum(ident)
        out["exact_match_excl_identity"] = round(100 * sum(em[k] for k in keep) / max(len(keep), 1), 2)
    return out


def copy_baseline(inputs, refs) -> dict:
    """Hypothesis = the dialect input itself. The 'do nothing' reference point."""
    return compute_all(list(inputs), list(refs), inputs=inputs)


# ---- neural metrics (lazy; GPU + downloads; untested in the sandbox) --------
_CACHE = {}   # models are loaded once per process and reused for every run folder

def comet_scores(srcs, mts, refs, batch_size=32, gpus=1):
    """wmt22-comet-da (paper 6.11). Returns (system_score, per_segment_scores)."""
    from comet import download_model, load_from_checkpoint
    if "comet" not in _CACHE:
        _CACHE["comet"] = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    model = _CACHE["comet"]
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return out.system_score, out.scores


def cometkiwi_scores(srcs, mts, batch_size=32, gpus=1):
    """Reference-free COMETKiwi-DA. The HF repo is gated: accept the licence and
    log in with an HF token first."""
    from comet import download_model, load_from_checkpoint
    if "kiwi" not in _CACHE:
        _CACHE["kiwi"] = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    model = _CACHE["kiwi"]
    data = [{"src": s, "mt": m} for s, m in zip(srcs, mts)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return out.system_score, out.scores


def bleurt_scores(mts, refs, batch_size=32, device="cuda"):
    """BLEURT-20 via the PyTorch port `bleurt-pytorch` (lucadiliello/BLEURT-20)."""
    import torch
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    name = "lucadiliello/BLEURT-20"
    if "bleurt" not in _CACHE:
        _CACHE["bleurt"] = (BleurtTokenizer.from_pretrained(name),
                            BleurtForSequenceClassification.from_pretrained(name).to(device).eval())
    tok, model = _CACHE["bleurt"]
    scores = []
    with torch.no_grad():
        for i in range(0, len(mts), batch_size):
            enc = tok(refs[i:i + batch_size], mts[i:i + batch_size], padding="longest",
                      truncation=True, max_length=512, return_tensors="pt").to(device)
            scores += model(**enc).logits.flatten().tolist()
    return sum(scores) / len(scores), scores
