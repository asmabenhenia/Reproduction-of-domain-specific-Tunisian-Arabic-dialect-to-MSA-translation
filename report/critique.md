# Critique (living document)

Every item has evidence that can be re-checked in this repo. "Verified" = reproduced by our code.

## A. Verified from the authors' files

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Validation set is a slice of the test set.** All 9 training scripts use `val = test_data[:80]` and an 85/15 split (2550/450). The paper describes a separate 150-pair validation set and 2400 training pairs. | e.g. `finetune_allam7b_tun_msa_r16_e3.py` lines 114-116, 149 |
| 2 | **LoRA targets differ from the paper.** Scripts adapt q,k,v,o,gate,up,down; paper (5.3) says `q_proj`, `v_proj` only. 4-bit QLoRA is never mentioned in the paper. | `target_modules` in all LLM scripts |
| 3 | **Headline run untraceable.** No script or result file for ALLaM Config 2 (BLEU 80.67). The provided `r16_e3` script has dropout .05, wd .01, no early stopping, i.e. not Table 5 Config 2. | file audit |
| 4 | **AraBART conflict.** Stored result: BLEU 62.49 / chrF++ 56.18 / TER 46.08. Paper Table 7: 74.79 / 69.39 / 31.11. The bootstrap section uses 62.49. Three AraBART scripts exist (lr 3e-5, 5e-5, 5e-4). | `original_results/arabart_evaluation_results.json` (**verified**: our code reproduces 62.49) |
| 5 | **Char-level BLEU inflates scores.** AraT5v2: 79.78 (char) vs 58.65 (13a). A copy-the-input baseline gets 57.37 char-BLEU / chrF++ 48.03 / TER 60.03 / EM 4.0%, and 21.17 with 13a. AraBART at 13a = 32.89. | `python -m tests.test_against_stored` |
| 6 | **Identity pairs.** 18/450 test pairs (4.0%) have input == reference; all 18 are in AraT5v2's 143 exact matches. EM excluding them: 28.94%. Corpus-wide 81 identity pairs, 34 duplicate outputs, 8 test outputs also in train. | `src/data.py audit` |
| 7 | **Seq2Seq and LLMs decoded differently** (beam 4 + `no_repeat_ngram_size=3` vs greedy), and AraT5v2 uses an undisclosed Arabic task prefix. | finetune_arat5v2 lines 118, 244-263 |
| 8 | **Token F1 is set-based**, not the multiset token-level F1 one would assume. Small effect (79.85 vs 79.67 for AraT5v2). | `src/metrics.py` |
| 9 | Corpus file differs slightly from what was evaluated (test item 0 reference corrected). | stored predictions vs corpus |
| 10 | Corpus instruction is French with 53 typo variants; paper shows an English instruction. | `src/data.py` |

## B. Verified: what is solid
Our independent code reproduces the authors' stored BLEU(char), chrF++, TER, Token F1 and Exact Match **exactly** for AraT5v2, AraBART and ALLaM Config 3, and the test set matches 450/450.

## C. To investigate with re-runs
* Does validation leakage matter? (compare checkpoint selection on leaked vs clean validation)
* Which AraBART learning rate gives 62.49 / 74.79?
* Is the LoRA-rank effect (r16 > r64) stable across seeds, given single-run, single-seed reporting?
* q/v-only LoRA (paper) vs all-linear LoRA (authors' code).
* ALLaM-vs-AraT5v2 gap: bootstrap CI includes zero in the paper itself; with 3 seeds is it distinguishable?
