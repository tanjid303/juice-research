"""Kaggle zip: Gemma-2-9B FP16 random-K10 suppress. No INT4 weights."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_fp16_random_k10_suppress"
ZIP = ROOT / "kaggle_juice_gemma2_fp16_random_k10_suppress.zip"

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

for rel in COMMON + [
    "scripts/gemma2_06_fp16_random_k10_suppress.py",
    "scripts/00_RUN_GEMMA2_06.py",
]:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)

shutil.copy2(ROOT / "scripts" / "00_RUN_GEMMA2_06.py", OUT / "00_RUN_GEMMA2_06.py")

(OUT / "data").mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "data" / "conflictqa_pilot250_popular.jsonl",
    OUT / "data" / "conflictqa_pilot250_popular.jsonl",
)

heads = OUT / "directional_heads"
heads.mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "juice_stage2_gemma2_base_cell3" / "random_k10_heads.json",
    heads / "random_k10_heads.json",
)
shutil.copy2(
    ROOT / "juice_stage2_gemma2_base_id" / "directional_heads" / "split.json",
    heads / "split.json",
)

nb = ROOT / "juice_repro" / "kaggle_gemma2_fp16_random_k10_suppress.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_fp16_random_k10_suppress.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_fp16_random_k10_suppress.ipynb")

readme = """# GEMMA2_06 — FP16 random-K10 suppress

**No INT4 checkpoint.** New notebook. Do **not** upload `_make_*.py`.

File → Upload `kaggle_gemma2_fp16_random_k10_suppress.ipynb`
or paste `00_RUN_GEMMA2_06.py` as the only cell.

First print must be `THIS IS GEMMA2_06`.

GPU T4 x2, Internet ON, HF_TOKEN. Attach this zip. Run All in the draft.

Same 10 heads as the INT4 suppress random control (Cell-3 enlarge draw, seed 42),
applied as H− (α+=0, α−=1). Same 50 ids, juice_coherent, bootstrap 1000 / seed 42.

```
/kaggle/working/gemma2_fp16_random_k10_suppress_summary.json
/kaggle/working/gemma2_fp16_random_k10_suppress_bootstrap.json
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
