"""Kaggle zip: Gemma-2-9B INT4 L27_H6 transfer (+ optional FP16 suppress). No requantize."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_int4_transfer_l27h6"
ZIP = ROOT / "kaggle_juice_gemma2_int4_transfer_l27h6.zip"

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
    "scripts/gemma2_02_quant_smoke_test.py",
    "scripts/gemma2_03_int4_transfer_l27h6.py",
    "scripts/gemma2_04_fp16_context_suppress_check.py",
]

HEADS_DIR = ROOT / "juice_stage2_gemma2_base_id" / "directional_heads"
HEAD_FILES = [
    "heads_dir_parametric_k1_raw.json",
    "heads_dir_context_k10.json",
    "split.json",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

py = str(ROOT / ".venv" / "Scripts" / "python.exe")
subprocess.check_call([py, str(ROOT / "juice_repro" / "_make_gemma2_int4_transfer_l27h6_nb.py")])
subprocess.check_call([py, str(ROOT / "juice_repro" / "_make_gemma2_fp16_context_suppress_nb.py")])

for rel in COMMON + SCRIPTS:
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
for name in HEAD_FILES:
    shutil.copy2(HEADS_DIR / name, heads / name)

fp16_ref = ROOT / "juice_stage2_gemma2_base_cell3" / "gemma2_base_cell3_bootstrap.json"
if fp16_ref.is_file():
    dst = OUT / "fp16_reference"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fp16_ref, dst / "gemma2_base_cell3_bootstrap.json")

(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
for nb_name in (
    "kaggle_gemma2_int4_transfer_l27h6.ipynb",
    "kaggle_gemma2_fp16_context_suppress.ipynb",
):
    nb = ROOT / "juice_repro" / nb_name
    shutil.copy2(nb, OUT / nb_name)
    shutil.copy2(nb, OUT / "juice_repro" / nb_name)

readme = """# Gemma-2-9B — INT4 transfer of L27_H6 (no requantize)

Weights are **not** in this zip. Attach the existing N_CALIB=128 checkpoint.

## Task 1 (priority)

Notebook: `kaggle_gemma2_int4_transfer_l27h6.ipynb`

1. GPU **T4 x2**, Internet ON, `HF_TOKEN`
2. Upload `kaggle_juice_gemma2_int4_transfer_l27h6.zip` as a dataset
3. **Add Input → Notebook Output Files** → gemma-quantize version with
   `gemma2_9b_base_gptq_int4_ncalib128/` (`config.json` + `*.safetensors`)
4. File → Upload the Task 1 notebook
5. Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.

Smoke (`gemma2_02`) then transfer (`gemma2_03`): L27_H6 K=1 + random-K1 (seed 42, exclude L27_H6).
Same 50 eval ids, `juice_coherent`, α+=1, α−=0, bootstrap 1000 / seed 42.

Download:

```
/kaggle/working/gemma2_int4_transfer_l27h6_summary.json
/kaggle/working/gemma2_int4_transfer_l27h6_bootstrap.json
```

Do **not** attach `gemma2_gptq_ncalib8`. Do **not** requantize.

## Task 2 (lower priority, FP16 only)

Notebook: `kaggle_gemma2_fp16_context_suppress.ipynb`

No quantized model. Same zip. New draft if Task 1 used the session.
Steers directional ctx K10 as H− (α+=0, α−=1).

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
