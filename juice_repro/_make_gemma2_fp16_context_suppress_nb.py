"""Generate juice_repro/kaggle_gemma2_fp16_context_suppress.ipynb"""
import json
from pathlib import Path

cells = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.split("\n")]})


def code(s: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [line + "\n" for line in s.split("\n")],
        }
    )


md(
    """# TASK 2 — Gemma-2-9B FP16 context K10 suppress

This **is** the run notebook. Do **not** upload `_make_gemma2_fp16_context_suppress_nb.py`.

**No INT4 checkpoint.** Loads `google/gemma-2-9b` FP16.

| Locked | Value |
|--------|--------|
| Heads | directional ctx K10 as **H−** (enlarge empty) |
| α | α+=**0**, α−=**1** |
| Eval | same 50 ids (`split.json`, seed 42) |
| Prompt | `juice_coherent`, greedy, 16 tokens |
| Stats | paired bootstrap 1000 / seed 42 |

Enlarge of this set was **−2 pp** (CI includes 0) at a 48% baseline. Suppress
asks whether that was no causal power, or a ceiling with no room to enlarge.

**~1–2 h** on GPU T4 ×2. **New notebook.** Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.
"""
)

md("## 0 — Locate script")
code(
    r"""import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

print("THIS IS TASK 2 — FP16 ctx K10 suppress. Not the _make_ generator.")
SCRIPT_04 = find_one("gemma2_04_fp16_context_suppress_check.py")
assert SCRIPT_04.name == "gemma2_04_fp16_context_suppress_check.py", SCRIPT_04
print("SCRIPT_04", SCRIPT_04)
find_one("heads_dir_context_k10.json")
find_one("split.json")
"""
)

md("## 1 — Suppress ctx K10 on FP16")
code(
    r"""print("=== TASK 2: gemma2_04_fp16_context_suppress_check ===")
exec(compile(SCRIPT_04.read_text(encoding="utf-8"), str(SCRIPT_04), "exec"))
print("=== suppress check done ===")
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "gpu",
    },
    "cells": cells,
}

try:
    _here = Path(__file__).resolve().parent
    _in_jupyter = False
except NameError:
    _here = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    _in_jupyter = True

out = _here / "kaggle_gemma2_fp16_context_suppress.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("Wrote", out)

if _in_jupyter:
    print("Detected Jupyter (no __file__). You uploaded the generator.")
    print("Running Task 2 script instead of stopping.")
    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    hits = []
    for root in roots:
        if root.exists():
            hits.extend(root.rglob("gemma2_04_fp16_context_suppress_check.py"))
    hits = [h for h in hits if h.is_file()]
    assert hits, (
        "Attach kaggle_juice_gemma2_fp16_context_suppress.zip "
        "(must contain scripts/gemma2_04_fp16_context_suppress_check.py)"
    )
    script = hits[0]
    print("SCRIPT_04", script)
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"))

