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
| 11 | **The authors' "validation BLEU" is not translation BLEU.** In the LLaMA Config 4 script, `compute_metrics` takes the argmax of the teacher-forced logits over the whole sequence (prompt + response), does not shift predictions against labels, and decodes both. About 80% of those characters are the fixed prompt. Illustration: a text compared with itself shifted by 4 characters scores char-BLEU 98.1 on the whole sequence but 89.8 on the response only. So (a) the 82-83 validation BLEU "from epoch 0" in Table 9 says little about translation quality; (b) the paper's reading that ALLaM's Arabic representations are "already largely aligned with MSA" (6.8) is not supported by that number; (c) "BLEU-based early stopping" selected checkpoints on a metric that is almost insensitive to translation errors; (d) the loss-vs-BLEU "decoupling" (6.7) may be an artefact. Caveat: read from the LLaMA C4 script; the ALLaM/Mistral v4000 scripts were not provided, so this is likely but unconfirmed for them. | `finetune_llama3_8b_tun_msa_v4r32.py` lines 221-260; sandbox illustration; we will measure it directly (`--es_impl authors_proxy`) |
| 12 | **AraT5v2 is not ~220M parameters.** The paper (Table 4, 6.2, 7.2) says ~220M and "about 30 times smaller" than the 7B LLM. The checkpoint the authors' script loads (`UBC-NLP/AraT5v2-base-1024`) has **367,508,736** parameters in our run, i.e. about 19x smaller than a 7B model, not 30x. (AraBART at ~139M is consistent.) | Kaggle dry run, `parameters:` line |

## B. Verified: what is solid
Our independent code reproduces the authors' stored BLEU(char), chrF++, TER, Token F1 and Exact Match **exactly** for AraT5v2, AraBART and ALLaM Config 3, and the test set matches 450/450.

## C. To investigate with re-runs
* Does validation leakage matter? (compare checkpoint selection on leaked vs clean validation)
* Which AraBART learning rate gives 62.49 / 74.79?
* Is the LoRA-rank effect (r16 > r64) stable across seeds, given single-run, single-seed reporting?
* q/v-only LoRA (paper) vs all-linear LoRA (authors' code).
* ALLaM-vs-AraT5v2 gap: bootstrap CI includes zero in the paper itself; with 3 seeds is it distinguishable?

## D. First re-run result (AraT5v2, seed 42, fp32, clean validation)
| | BLEU(char) | chrF++ | TER | TokenF1 | EM |
|---|---|---|---|---|---|
| Paper / authors' stored | 79.78 | 74.96 | 25.32 | 79.85 | 31.78 |
| Ours (ckpt step 200) | 76.21 | 71.14 | 29.14 | 76.55 | 24.89 |

* Paired bootstrap on the same 450 test sentences: authors - ours = 3.56 BLEU, 95% CI [2.59, 4.57], authors better in 100% of resamples. The gap is real for these two runs (this CI covers test-sentence sampling only, not training randomness).
* Exact matches: authors 143, ours 112; only-authors 40, only-ours 9. Typical failure in ours: **omission** of a phrase (e.g. dropping "لغير المسلمين").
* Our validation loss is lowest at step 300 but the kept checkpoint is step 200 (eval every 100 / save every 200, the authors' setting): the best evaluated model is not the one kept.
* Candidate causes, to be separated by controlled runs: (1) checkpoint alignment (`--save_steps 100`); (2) the authors' protocol: 2550 training sentences and validation = test[:80]; (3) bf16 weights vs our fp32; (4) seed variance.

### D2. Ablation results (AraT5v2, seed 42, same 450 test sentences)
| Run | What changes | BLEU(char) | chrF++ | TER | EM | kept ckpt |
|---|---|---|---|---|---|---|
| First run | paper split, eval 100 / save 200 (authors' setting) | 76.21 | 71.14 | 29.14 | 24.89 | step 200 |
| A | paper split, save every 100 | **78.21** | 73.00 | 27.65 | 27.11 | step 300 |
| B | authors' protocol (2550 train, val = test[:80]), eval 100 / save 200 | **78.46** | 73.30 | 27.34 | 27.78 | step 200 |
| Paper | | 79.78 | 74.96 | 25.32 | 31.78 | n/a |

* The checkpoint-saving mismatch alone costs about 2.0 BLEU (A vs first run).
* B mixes two changes: +150 training sentences and a validation set that overlaps the test set by 80 sentences. It still kept the step-200 checkpoint. Not yet separated; combining B with save-every-100 was not run.
* Remaining gap to the paper after A: 1.57 BLEU, single seed; seed spread unmeasured.

## E. AraBART learning-rate sweep (seed 42, fp32, paper protocol, same 450 test sentences)
| lr | BLEU(char) | chrF++ | TER | EM | kept ckpt |
|---|---|---|---|---|---|
| 5e-4 (paper's stated lr) | 73.11 | 67.16 | 34.40 | 21.11 | step 100 (training diverged early) |
| 5e-5 | 75.41 | 70.36 | 29.62 | 25.56 | step 400 |
| 3e-5 | **75.47** | 70.26 | 29.90 | 25.56 | step 600 |
| Paper Table 7 | 74.79 | 69.39 | 31.11 | n/a | n/a |
| Authors' stored file | 62.49 | 56.18 | 46.08 | n/a | n/a |

* **None of our three learning rates reproduces the stored 62.49** (our worst is 73.11).
* **The paper's narrative is not supported:** 6.2 and 7.1 present 5e-4 as the rescue ("confirming the sensitivity of BART-based models to learning rate selection"). In our runs 5e-4 is the *worst* rate (best checkpoint at step 100) and 3e-5 / 5e-5 are 2.3 BLEU better and equally good. We see no strong sensitivity among the low rates.
* **Hypothesis (untested yet): the authors' low-lr result came from pure-bf16 weights.** Their scripts load the model with `torch_dtype=bfloat16`. With bf16 weights (about 3 significant digits) Adam updates of size ~5e-5 can fall below the representable step for many weights, so training stalls; 5e-4 survives. That would turn an artefact of numeric precision into a reported "learning-rate sensitivity". Test: `--weights_dtype bf16` at lr 5e-5 vs 5e-4.
* **Consequence for the headline comparison:** the paper's significance claim "ALLaM-7B vs AraBART +18.15 BLEU" is built on 62.49. Against our AraBART (75.4) the gap to AraT5v2-class systems is about 3-5 BLEU, not 17-18. The human-evaluation AraBART outputs (Table 11) come from an unknown run.
* **Corpus bytes:** the corpus file in the GitHub repo has a different SHA-256 than the original (0d958b55... vs 417a6b68...), i.e. it was byte-modified (likely CRLF conversion). Content parses identically (all tests pass); `.gitattributes` now prevents further conversion.

## F. Qualitative look at our AraBART outputs (seed 42, same test set)
* **The paper's two AraBART anecdotes do not reproduce.** The "worship vs apostasy" reversal: our test item 13 (Yaqub's will) is translated correctly by all three AraBART runs. The "milk brother -> inheritance" hallucination: item 16 (`خويا مرضاعة`) is answered with a rendering that loses the milk-kinship concept ("أخوتي من أمي وبويا") rather than an inheritance question.
* **Failure modes we do see:** (a) loss of the legal key concept (item 16); (b) entity confusion (item 448: "ابنتي رضعت من أبي", nonsense for a milk-kinship question; AraT5v2 gets it right); (c) **copying the dialect input unchanged** (item 370 for lr 3e-5 and 5e-5), which char-BLEU rewards because of the 57.4 copy baseline.
* A crude screen for topic hallucination (output shares <20% of tokens with both source and reference) flags only 1-3 of 450 sentences per run, so unrelated-topic outputs are rare in our runs. Not comparable to the paper's 20% critical-error rate (which also counts omissions and entity errors); needs a proper annotation.
* Run-to-run variation is large at sentence level: AraBART lr 3e-5 and 5e-5 give identical output for only 283 of 450 sentences, though their BLEU differs by 0.06.
