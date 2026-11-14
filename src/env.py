"""env.py -- print library versions at the start of a run and fail early on known-bad ones."""


def check_env():
    import importlib
    vers = {}
    for pkg in ("torch", "transformers", "tokenizers", "peft", "bitsandbytes", "accelerate", "sacrebleu"):
        try:
            vers[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            vers[pkg] = "not installed"
    print("env:", ", ".join(f"{k}={v}" for k, v in vers.items()), flush=True)
    tf = vers["transformers"]
    if tf != "not installed" and int(tf.split(".")[0]) >= 5:
        raise RuntimeError(
            f"transformers {tf} cannot load the AraT5v2 tokenizer (T5 Unigram 'vocab' TypeError). "
            'Run:  pip install -q "transformers>=4.46,<5"   then re-run (no kernel restart needed '
            "for !python commands).")
    return vers
