# DECISIONS.md

Source of truth for every choice **not fixed by the paper** and every **deviation** from it.
Tags: **[paper]** follows the paper, **[authors' code]** follows the authors' scripts where the
paper is silent, **[forced]** hardware constraint, **[ours]** our own choice.

| ID | Topic | Decision | Tag | Why |
|----|-------|----------|-----|-----|
| D1 | Split | 2400 train / 150 val / 450 test. Same shuffle (seed 42) as the authors, so the **test set is identical** (last 450). Val = items 2400-2549, disjoint from test. | paper + ours | The authors' scripts used 85/15 and `val = test[:80]` (validation leaked from test). |
| D2 | Model IDs | `humain-ai/ALLaM-7B-Instruct-preview`, `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `UBC-NLP/AraT5v2-base-1024`, `moussaKam/AraBART` | authors' code | Paper names families only. |
| D3 | Quantisation | LLMs loaded in 4-bit NF4 + double quant (QLoRA), including zero-shot | forced + authors' code | 16 GB T4. Paper never mentions quantisation. |
| D4 | LoRA targets | `q_proj`, `v_proj` only, alpha = 2r, bias none | **paper** | Authors' code adapted all 7 linear layers. Optional extra run: all-linear ALLaM C2 to measure the gap. |
| D5 | Prompts | Fine-tuning/inference: Alpaca-style `### Instruction/Input/Response`. Zero-shot: each model's native format. | authors' code | Paper: "format of Sec. 4.3" only. |
| D6 | Token F1 / EM | Token F1 = set-based, whitespace tokens, mean per sentence (multiset variant also reported). EM = equal after stripping trailing `،` and `.` | authors' code | Paper gives no formula. |
| D7 | AraT5v2 prefix | `ترجم من اللهجة التونسية إلى العربية الفصحى: ` | authors' code | Not in paper. |
| D8 | Seq2Seq decoding | beam 4, early_stopping, `no_repeat_ngram_size=3` (LLMs: greedy). Differs from LLM decoding. | authors' code | Not in paper. |
| D9 | Precision | LoRA: fp16 compute. Seq2Seq: fp32 weights and training. | forced | T4 has no bf16. Authors loaded seq2seq weights in bf16 and trained in pure bf16 (`torch_dtype=bfloat16`), which can change results; T5 is unstable in fp16. |
| D10 | Instruction field | The 53 typo variants (e.g. `tunسien`) are replaced by the majority instruction. French text kept. | ours | Harmless but untidy; paper shows an English instruction, corpus is French. |
| D11 | Optimiser etc. | lr 3e-4 [paper]; `paged_adamw_8bit`, cosine, warmup 100, batch 4x4=16, max_seq 512 | authors' code | Paper says "AdamW". |
| D12 | Early-stopping metric | Configs 2/3/4: **generation-based** val char-BLEU (greedy decode of the 150 val sentences, vs references), patience 1, checkpoint chosen on it. Config 1: none (best checkpoint by val loss). Seq2Seq: val loss, patience 3. | paper + ours | Paper: "BLEU-based early stopping". The authors' implementation is a different, much weaker quantity (see D19). |
| D13 | Corrected reference | The released corpus has one test reference corrected after the authors' evaluation (المفروم -> المفرم, test item 0). We use the released corpus. | ours | Effect on corpus metrics is negligible. |
| D14 | ALLaM "Config 2" | Built from paper Table 5 (r16, dropout .10, wd .10, patience 1, 3 ep). The authors' `finetune_allam7b_tun_msa_r16_e3.py` is NOT this config (dropout .05, wd .01, no ES). | paper | Their Config 2 script was not provided. |
| D15 | Reconstructed runs | ALLaM C2/C3, LLaMA C2, Mistral C2 have no authors' script; built from Table 5 only. | ours | Missing files. |
| D16 | Per-sentence generation for LLMs | Causal-LM decoding runs one sentence at a time (batch 1), as the authors did. | authors' code | With batching, left-padding uses the EOS token as padding; `repetition_penalty=1.1` then penalises EOS and changes output length. |
| D17 | Multi-GPU | Training is pinned to one GPU (`CUDA_VISIBLE_DEVICES=0`) so effective batch = 32 as in Table 4. The authors' script scales the batch with the number of GPUs. | ours | Kaggle T4 x2 would silently double the batch. |
| D18 | LLM training text | Alpaca text, **explicit EOS id appended**, default special tokens (BOS) as at inference, loss over the whole sequence (prompt+response) as in the authors' SFT setup. Custom collator keeps the EOS label even though pad == eos. | authors' code + ours | HF's language-model collator masks every pad-id token, which silently removes EOS supervision when pad == eos. |
| D19 | Validation BLEU implementation | Default `metric_impl: generation`. Option `authors_proxy` reproduces the authors' `compute_metrics` (argmax of teacher-forced logits over the whole sequence, unshifted). Validation decoding is batched, left-padded, repetition_penalty 1.0 (speed); test decoding uses the paper settings. | ours | Lets us run C2 both ways and measure the effect on checkpoint selection and test score. |
| D20 | Gated models | `HF_TOKEN` read from the environment (Kaggle Secrets); LLaMA 3 licence must be accepted on Hugging Face. | forced | |
