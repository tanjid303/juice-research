"""Tiny Kaggle zip: rebuild the lost N_CALIB=128 GPTQ checkpoint. Quantize only."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_int4_ncalib128_quantize_only"
ZIP = ROOT / "kaggle_juice_stage2_int4_ncalib128_quantize_only.zip"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "juice_repro" / "_make_ncalib128_quantize_only_nb.py"),
    ]
)

dst = OUT / "scripts"
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "scripts" / "ncalib128_01_quantize.py", dst / "ncalib128_01_quantize.py")

nb_src = ROOT / "juice_repro" / "kaggle_stage2_int4_ncalib128_quantize_only.ipynb"
shutil.copy2(nb_src, OUT / "kaggle_stage2_int4_ncalib128_quantize_only.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, OUT / "juice_repro" / "kaggle_stage2_int4_ncalib128_quantize_only.ipynb")

readme = """# Rebuild the lost N_CALIB=128 GPTQ checkpoint

Quantize **only**. Does not re-run transfer cells.

Output: `/kaggle/working/llama31_8b_base_gptq_int4_ncalib128/` (~4–6 GB)

## Run

1. Upload this zip as a Kaggle dataset
2. **File → Upload notebook** `kaggle_stage2_int4_ncalib128_quantize_only.ipynb`
3. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
4. Accept the Llama-3.1-8B license on Hugging Face
5. Save & Run All (~45–90 min)
6. When it prints **SAVE VERSION NOW**, click Save Version immediately

Then open the native-clean-K2 notebook and **Add Input → Notebook Output Files → this Save Version**.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
(OUT / "KAGGLE.md").write_text(readme, encoding="utf-8")

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(OUT).as_posix())

print("Wrote", OUT)
print("Wrote", ZIP)
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT), path.stat().st_size)
