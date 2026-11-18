"""Kaggle zip: Gemma-2-9B FP16 context K10 suppress only. No INT4 weights."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_fp16_context_suppress"
ZIP = ROOT / "kaggle_juice_gemma2_fp16_context_suppress.zip"

COMMON = [
    "common/__init__.py",
    "common/models.py",
    "common/juice.py",
    "common/juice_attn.py",
    "common/conflictqa.py",
    "common/scoring.py",
    "common/stats_analysis.py",
    "common/inference.py",
    "common/prompts.py",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_gemma2_fp16_context_suppress_nb.py")]
)

for rel in COMMON + ["scripts/gemma2_04_fp16_context_suppress_check.py"]:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)

(OUT / "data").mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "data" / "conflictqa_pilot250_popular.jsonl",
    OUT / "data" / "conflictqa_pilot250_popular.jsonl",
)

heads = OUT / "directional_heads"
heads.mkdir(parents=True, exist_ok=True)
src_heads = ROOT / "juice_stage2_gemma2_base_id" / "directional_heads"
shutil.copy2(src_heads / "heads_dir_context_k10.json", heads / "heads_dir_context_k10.json")
shutil.copy2(src_heads / "split.json", heads / "split.json")

nb = ROOT / "juice_repro" / "kaggle_gemma2_fp16_context_suppress.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_fp16_context_suppress.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_fp16_context_suppress.ipynb")

readme = """# TASK 2 — Gemma-2-9B FP16 context K10 suppress

**One notebook only.** File → Upload `kaggle_gemma2_fp16_context_suppress.ipynb`.
Do not upload `_make_*.py`.

No INT4 checkpoint. New Kaggle notebook, GPU T4 x2, Internet ON, HF_TOKEN.

1. Upload `kaggle_juice_gemma2_fp16_context_suppress.zip` as a dataset
   (or reuse the Task 1 zip dataset if it already has `gemma2_04_fp16_context_suppress_check.py`)
2. File → Upload `kaggle_gemma2_fp16_context_suppress.ipynb`
3. First code cell must print: `THIS IS TASK 2 — FP16 ctx K10 suppress`
4. Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.

α+=0, α−=1, enlarge empty, same 50 eval ids, juice_coherent, bootstrap 1000 / seed 42.

```
/kaggle/working/gemma2_fp16_context_suppress_summary.json
/kaggle/working/gemma2_fp16_context_suppress_bootstrap.json
```
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
(OUT / "KAGGLE.md").write_text(readme, encoding="utf-8")

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(OUT).as_posix())

print("Wrote", ZIP, "bytes", ZIP.stat().st_size)
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT), path.stat().st_size)
