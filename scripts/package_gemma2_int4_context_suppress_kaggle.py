"""Kaggle zip: Gemma-2-9B INT4 context-K10 suppress transfer. No requantize."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_int4_context_suppress"
ZIP = ROOT / "kaggle_juice_gemma2_int4_context_suppress.zip"

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

SCRIPTS = [
    "scripts/gemma2_ncalib128_quantize.py",
    "scripts/gemma2_02_quant_smoke_test.py",
    "scripts/gemma2_03_int4_transfer_l27h6.py",
    "scripts/gemma2_04_fp16_context_suppress_check.py",
    "scripts/gemma2_05_int4_context_suppress_transfer.py",
    "scripts/00_RUN_GEMMA2_05.py",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

for rel in COMMON + SCRIPTS:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)

# Zip-root runner: first file alphabetically if someone uploads a .py
shutil.copy2(ROOT / "scripts" / "00_RUN_GEMMA2_05.py", OUT / "00_RUN_GEMMA2_05.py")

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
shutil.copy2(
    ROOT / "juice_stage2_gemma2_base_cell3" / "random_k10_heads.json",
    heads / "random_k10_heads.json",
)

fp16_boot = ROOT / "task-2-res" / "juice_stage2_gemma2_fp16_context_suppress" / "gemma2_fp16_context_suppress_bootstrap.json"
if not fp16_boot.is_file():
    fp16_boot = ROOT / "task-2-res" / "gemma2_fp16_context_suppress_bootstrap.json"
if fp16_boot.is_file():
    ref = OUT / "fp16_reference"
    ref.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fp16_boot, ref / "gemma2_fp16_context_suppress_bootstrap.json")

nb = ROOT / "juice_repro" / "kaggle_gemma2_int4_context_suppress.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_int4_context_suppress.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_int4_context_suppress.ipynb")

readme = """# GEMMA2_05 — INT4 context K10 suppress transfer

No requantize. Attach the existing N_CALIB=128 checkpoint.

## Upload

**File → Upload notebook** `kaggle_gemma2_int4_context_suppress.ipynb`

Or paste `00_RUN_GEMMA2_05.py` as the only cell.

Do **not** upload `_make_*.py`. First print must be `THIS IS GEMMA2_05`.

## Attach

1. This zip as a dataset
2. **Add Input → Notebook Output Files** → gemma-quantize version with
   `gemma2_9b_base_gptq_int4_ncalib128/` (`config.json` + `*.safetensors`)

GPU T4 x2, Internet ON, HF_TOKEN. **New notebook.** Run All in the draft.
Quick Save when it prints `SAVE VERSION NOW`.

Scripts 01–04 are in `scripts/` for reference. This notebook only runs 05.

Random-K10 control reuses the Gemma FP16 Cell-3 enlarge draw (seed 42),
applied as H−. Not a new sample.

```
/kaggle/working/gemma2_int4_context_suppress_summary.json
/kaggle/working/gemma2_int4_context_suppress_bootstrap.json
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
