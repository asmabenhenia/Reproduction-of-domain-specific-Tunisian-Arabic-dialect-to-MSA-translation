"""Run: python -m tests.test_generate_io   (no torch needed)
Checks save_run writes predictions/metrics and that a perfect prediction scores perfectly."""
import json, os, tempfile
from src import data, generate

test = data.load_split("data/splits", "test")[:20]
with tempfile.TemporaryDirectory() as d:
    res = generate.save_run(d, test, [x["output"] for x in test], meta={"t": 1})
    m = res["metrics"]
    assert m["bleu_char"] == 100.0 and m["exact_match"] == 100.0 and m["ter"] == 0.0, m
    rows = generate.load_predictions(os.path.join(d, "predictions.jsonl"))
    assert len(rows) == 20 and rows[0]["reference"] == test[0]["output"]
    assert os.path.exists(os.path.join(d, "metrics.json"))
print("test_generate_io: OK")
