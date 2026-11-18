"""Kaggle zip: Gemma-2-9B JuICE FP16 wrap + tiny GPTQ smoke."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_smoke"
ZIP = ROOT / "kaggle_juice_gemma2_smoke.zip"

COMMON = [
    "common/__init__.py",
    "common/models.py",
    "common/juice.py",
    "common/juice_attn.py",
    "common/conflictqa.py",
    "common/scoring.py",
    "common/prompts.py",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_gemma2_smoke_nb.py")]
)

for rel in COMMON + ["scripts/gemma2_juice_smoke.py"]:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)

(OUT / "data").mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "data" / "conflictqa_pilot250_popular.jsonl",
    OUT / "data" / "conflictqa_pilot250_popular.jsonl",
)

nb = ROOT / "juice_repro" / "kaggle_gemma2_juice_smoke.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_juice_smoke.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_juice_smoke.ipynb")

readme = """# Gemma-2-9B JuICE smoke (Kaggle T4 x2)

FP16 wrap + dual-run, then 1-layer / 8-sample GPTQ decode check.

**Do not** start N_CALIB=128 until GPTQ generations are real words.
**Do not** use `/kaggle/working/gemma2_gptq_tiny_layer0` as a Stage 2 checkpoint.

## Run

1. Upload `kaggle_juice_gemma2_smoke.zip` as a Kaggle dataset
2. **File → Upload notebook** `kaggle_gemma2_juice_smoke.ipynb`
3. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
4. Accept [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) on Hugging Face
5. Save & Run All (~20–40 min)

Pass: config 256 / 16 / 8 / 50, 42 wrapped layers, shape `[B, S, 3584]`, GPTQ gens not garbage.
Fail (garbage decode or o_proj wrap): pivot to Qwen2.5-7B.
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
print("Wrote", ZIP, "bytes", ZIP.stat().st_size)
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT), path.stat().st_size)
