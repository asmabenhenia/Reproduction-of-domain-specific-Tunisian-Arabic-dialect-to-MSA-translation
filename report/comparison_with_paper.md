# Comparison with the paper (filled as experiments finish)

Columns: **Paper** = value printed in the paper, **Stored** = authors' result file, **Ours** = our re-run.
"n/a" = no source. Test set = the same 450 sentences everywhere.

## Table 7 -- Seq2Seq (BLEU char / chrF++ / TER)
| System | Paper | Stored | Ours |
|---|---|---|---|
| AraT5v2 | 79.78 / 74.96 / 25.32 | 79.78 / 74.96 / 25.32 | **76.21 / 71.14 / 29.14** (EM 24.89, TokenF1 76.55; ckpt step 200, seed 42, fp32) |
| AraBART | 74.79 / 69.39 / 31.11 | **62.49 / 56.18 / 46.08** | _todo_ (lr 3e-5, 5e-5, 5e-4) |

## Table 6 -- zero-shot (BLEU / chrF++ / TER)
| System | Paper | Stored | Ours |
|---|---|---|---|
| ALLaM-7B | 24.04 / 27.83 / 87.71 | n/a | _todo_ |
| LLaMA 3-8B | 5.42 / 15.59 / 799.83 | n/a | _todo_ |
| Mistral-7B | 9.73 / 20.93 / 376.71 | n/a | _todo_ |

## Table 8 -- LoRA (BLEU / chrF++ / TER / TokenF1 / EM)
| Model | Config | Paper | Stored | Ours |
|---|---|---|---|---|
| ALLaM-7B | C2 r16 3ep | 80.67 / 75.98 / 26.33 / 79.55 / 32.22 | n/a | _todo_ |
| ALLaM-7B | C3 r16 5ep | 79.88 / 75.59 / 27.59 / 79.33 / 31.11 | 79.88 / 75.59 / 27.59 / 79.33 / 31.11 | _todo_ |
| ALLaM-7B | C1 r64 4ep | 77.77 / 72.89 / 30.60 / 76.35 / 25.11 | n/a | _todo_ |
| LLaMA 3-8B | C4 r32 2ep | 76.05 / 70.71 / 32.04 / 74.95 / 25.56 | n/a | _todo_ |
| LLaMA 3-8B | C2 r16 3ep | 76.26 / 70.07 / 32.41 / 74.37 / 23.56 | n/a | _todo_ |
| LLaMA 3-8B | C1 r64 4ep | 76.07 / 69.80 / 31.87 / 74.34 / 24.44 | n/a | _todo_ |
| Mistral-7B | C2 r16 3ep | 75.35 / 69.15 / 32.77 / 74.24 / 25.33 | n/a | _todo_ |
| Mistral-7B | C1 r64 4ep | 73.18 / 66.60 / 35.64 / 71.74 / 21.11 | n/a | _todo_ |

## Table 12 -- neural metrics (COMET / COMETKiwi-DA / BLEURT)
| System | Paper | Ours |
|---|---|---|
| ALLaM-7B C2 | 0.9382 / 0.6125 / 0.7901 | _todo_ |
| AraT5v2 | 0.9293 / 0.6137 / 0.7830 | _todo_ |
| AraBART | 0.8487 / 0.5452 / 0.6241 | _todo_ |

## Table 13 -- beam 4 vs greedy (ALLaM C2): BLEU 80.67 -> 69.38; EM 32.22 -> 6.89; human 96.0 -> 93.0
