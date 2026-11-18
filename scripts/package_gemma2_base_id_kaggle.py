"""Kaggle zip: Gemma-2-9B base FP16 JuICE ID."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_base_id"
ZIP = ROOT / "kaggle_juice_gemma2_base_id.zip"

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
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_gemma2_base_id_nb.py")]
)

for rel in COMMON:
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
shutil.copy2(
    ROOT / "juice_stage2_base_id" / "directional_heads" / "split.json",
    heads / "split.json",
)

nb = ROOT / "juice_repro" / "kaggle_gemma2_base_id.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_base_id.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_base_id.ipynb")

readme = """# Gemma-2-9B base JuICE ID (FP16)

Does not use the INT4 checkpoint. Same 4 calib IDs as Llama Stage 2.

1. New notebook, GPU T4 x2, HF_TOKEN
2. Upload `kaggle_juice_gemma2_base_id.zip`
3. File → Upload `kaggle_gemma2_base_id.ipynb`
4. Run All in the **draft** (~3–5 h). Do not Save Version first.
5. Quick Save when it prints SAVE VERSION NOW.

Need: `juice_stage2_gemma2_base_id/directional_heads/heads_dir_context_k10.json`
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
