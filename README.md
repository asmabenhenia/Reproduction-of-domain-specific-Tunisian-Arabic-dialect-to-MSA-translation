# tun_msa_repro

Reproduction of *"Domain-specific Tunisian Arabic dialect-to-MSA translation in Islamic jurisprudence:
a multi-paradigm benchmarking of Seq2Seq models and LoRA-adapted LLMs"* (JKSUCIS 2026, 38:959),
built from the **paper's** specification, not from the authors' repository.

* `DECISIONS.md` lists every choice the paper leaves open and every deviation.
* Each folder `results/tableN_*` matches one table of the paper.
* `original_results/` holds the authors' stored outputs, used to validate our metric code.

## Quick start
```bash
pip install -r requirements-local.txt   # on your PC (light, no GPU/GitHub needed)
# pip install -r requirements.txt       # on Kaggle, for training
python -m src.data                      # writes data/splits/{train,val,test}.json
python -m tests.test_against_stored     # must print ALL CHECKS PASSED
```

## Paper table -> experiment -> config
| Paper | Experiment | Config |
|---|---|---|
| Table 6 | zero-shot, 3 LLMs | `configs/zeroshot.yaml` |
| Table 7 | AraT5v2, AraBART | `configs/seq2seq_*.yaml` |
| Table 8/9 | 8 LoRA runs | `configs/lora_<model>_c<n>.yaml` |
| Table 10 | best of each paradigm | built by analysis from the above |
| Table 11 | human evaluation | `src/stats.py` on the expert files |
| Table 12 | COMET / COMETKiwi / BLEURT | `src/metrics.py` neural functions |
| Table 13 | beam search | `configs/beam_allam7b_c2.yaml` |

## Status
| Module | State |
|---|---|
| `src/data.py`, `src/metrics.py` (lexical), `tests/` | done, validated against stored results |
| `src/metrics.py` neural functions | written, not yet run (needs GPU/internet) |
| `src/train_seq2seq.py`, `src/generate.py` | written, **not yet run on a GPU** (first Kaggle run = real test) |
| `src/train_lora.py` | written, unit-tested for data prep only, **not yet run on a GPU** |
| zero-shot + beam CLI, `stats.py` | next |

## Kaggle: run AraT5v2 (cell by cell)
```python
!git clone https://github.com/<you>/tun_msa_repro.git && cd tun_msa_repro && pip install -q -r requirements.txt
%cd tun_msa_repro
!python -m src.data
!python -m tests.test_against_stored          # must print ALL CHECKS PASSED
!python -m src.train_seq2seq --config configs/seq2seq_arat5v2.yaml --dry_run   # ~1-2 min smoke test
!python -m src.train_seq2seq --config configs/seq2seq_arat5v2.yaml             # real run
```

## Kaggle: LoRA dry run, then ALLaM Config 2
```python
import os; os.environ["HF_TOKEN"] = "hf_..."      # or use Kaggle Secrets; needed for LLaMA 3 only
!python -m src.train_lora --config configs/lora_allam7b_c2.yaml --dry_run     # smoke test
!python -m src.train_lora --config configs/lora_allam7b_c2.yaml               # real run
!python -m src.train_lora --config configs/lora_allam7b_c2.yaml --es_impl authors_proxy   # their val-BLEU proxy
```
