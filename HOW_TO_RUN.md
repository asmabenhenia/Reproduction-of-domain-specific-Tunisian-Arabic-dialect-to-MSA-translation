# HOW TO RUN (simple version)

## Once: put the repo on GitHub
1. Unzip `tun_msa_repro.zip` **into your repo folder** (the folder that contains `src`, `configs`, `data`). Replace files if asked.
2. In PowerShell, inside that folder:
   ```
   git add .
   git commit -m "final version"
   git push
   ```

## Every Kaggle session (sessions 1 to 4)
1. Kaggle -> **File -> Import Notebook** -> `kaggle_runner.ipynb` (also in `notebooks/`).
2. Settings (right panel): **Accelerator = GPU T4 x2**, **Internet = On**.
3. In cell 1 set `SESSION = 1` (then 2, 3, 4 in later sessions).
4. Click **Save Version -> Save & Run All**. Close the browser if you like.
5. When the version shows *Complete*: open the notebook page -> **Output** tab -> download `results_session<N>.zip`.
6. Upload that zip to the chat.

| SESSION | Jobs | Time |
|---|---|---|
| 1 | ALLaM config 2, beam re-run, ALLaM config 3, ALLaM config 1 | ~3.5 h |
| 2 | AraT5v2 final run, Mistral config 2, Mistral config 1 | ~2.5 h |
| 3 | LLaMA 3 configs 2, 4, 1 (needs HF token + Llama 3 licence) | ~3.5 h |
| 4 | Zero-shot ALLaM, Mistral, LLaMA 3 (needs HF token) | ~3 h |
| 5 | Neural metrics (COMET / COMETKiwi / BLEURT) on all runs | ~0.5 h |

Session 5 only works after all results are in the repo on GitHub (the chat will send you the merged repo).

## Hugging Face token (sessions 3, 4, 5)
huggingface.co -> accept the licence on `meta-llama/Meta-Llama-3-8B-Instruct` (and `Unbabel/wmt22-cometkiwi-da` for session 5)
-> Settings -> Access Tokens -> create a *Read* token -> in Kaggle: Add-ons -> Secrets -> new secret named `HF_TOKEN`
(attach it to the notebook). Never paste the token in a cell.

## Quick test before a long run
Set `DRY_RUN = True` in cell 1 and run the notebook normally: every job runs on a handful of sentences (a few minutes). Set it back to `False` for the real run.

## If something goes wrong
Copy the last 40 lines of the failing cell (or of the saved version's log) into the chat.

## Optional: make the tables and plots yourself (on your PC, no GPU)
```
pip install -r requirements-local.txt
python -m src.plot_curves      # curves.png in every run folder + results/table9_training_dynamics/table9.md
python -m src.make_report      # report/results_tables.md (paper vs ours, bootstrap significance tests)
```
