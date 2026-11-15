"""Run: python -m tests.test_protocols   (needs data/splits and data/splits_authors)"""
from src import data
raw = data.load_raw("data/raw/dataset_final_3000.json")
tr, va, te = data.make_splits(raw)
atr, ava, ate = data.make_splits_authors(raw)
assert te == ate, "test set must be identical under both protocols"
assert (len(tr), len(va), len(te)) == (2400, 150, 450) and (len(atr), len(ava), len(ate)) == (2550, 80, 450)
assert ava == ate[:80], "authors' validation is a slice of the test set"
assert not ({d["input"] for d in va} & {d["input"] for d in te}), "paper-protocol val must be disjoint from test"
assert tr == atr[:2400]
print("test_protocols: OK  (authors' val overlaps test:", len(ava), "items; ours overlaps: 0)")
