"""Run: python -m tests.test_lora_prep   (no torch needed; uses a stub tokenizer)"""
from src import data
from src.train_lora import pad_batch, tokenize_items


class StubTok:
    eos_token_id = 2          # pad == eos on purpose (ALLaM/LLaMA/Mistral situation)
    def __call__(self, text, truncation=True, max_length=512):
        ids = [1] + [100 + (ord(c) % 50) for c in text]      # 1 = BOS
        return {"input_ids": ids[:max_length]}


items = data.load_split("data/splits", "val")[:3]
rows = tokenize_items(StubTok(), items, max_len=512)
for r in rows:
    assert r["input_ids"][0] == 1 and r["input_ids"][-1] == 2, "BOS first, EOS last"

b = pad_batch(rows, pad_id=2)
L = len(b["input_ids"][0])
assert all(len(x) == L for x in b["input_ids"]) and all(len(x) == L for x in b["labels"])
for ids, att, lab, r in zip(b["input_ids"], b["attention_mask"], b["labels"], rows):
    n = len(r["input_ids"])
    assert lab[n - 1] == 2, "real EOS must stay supervised although pad_id == eos_id"
    assert all(v == -100 for v in lab[n:]) and all(v == 0 for v in att[n:])
    assert lab[:n] == r["input_ids"]
# truncation keeps EOS
long = tokenize_items(StubTok(), items[:1], max_len=20)[0]["input_ids"]
assert len(long) == 20 and long[-1] == 2
print("test_lora_prep: OK")
