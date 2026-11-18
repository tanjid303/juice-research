"""Kaggle zip: Gemma-2-9B N_CALIB=128 GPTQ, quantize only."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_ncalib128"
ZIP = ROOT / "kaggle_juice_gemma2_ncalib128.zip"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_gemma2_ncalib128_nb.py")]
)

dst = OUT / "scripts"
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "scripts" / "gemma2_ncalib128_quantize.py", dst / "gemma2_ncalib128_quantize.py")

nb = ROOT / "juice_repro" / "kaggle_gemma2_ncalib128_quantize.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_ncalib128_quantize.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_ncalib128_quantize.ipynb")

readme = """# Gemma-2-9B N_CALIB=128 GPTQ (quantize only)

Canonical INT4 for Gemma Stage 2. Not the N_CALIB=8 smoke.

Output: `/kaggle/working/gemma2_9b_base_gptq_int4_ncalib128/` (~5–7 GB)

## Run

1. **New** notebook — do not reuse the smoke notebook
2. Upload `kaggle_juice_gemma2_ncalib128.zip` as a dataset
3. File → Upload notebook `kaggle_gemma2_ncalib128_quantize.ipynb`
4. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
5. Accept google/gemma-2-9b
6. **Run All in the draft** (~60–120 min). Do **not** click Save Version first.
7. When it prints **SAVE VERSION NOW**, **Quick Save** (not Save & Run All).
   Failed Save & Run All wipes the checkpoint (Gemma Version 2).
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
