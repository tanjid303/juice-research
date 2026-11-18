"""Assemble kaggle upload folder + zip for base Cell-3 directional steering."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_base_cell3"
ZIP = ROOT / "kaggle_juice_stage2_base_cell3.zip"

COMMON_FILES = [
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

HEADS_DIR = ROOT / "juice_stage2_base_id" / "directional_heads"
HEAD_FILES = [
    "heads_dir_context_k10.json",
    "heads_dir_context_k5.json",
    "heads_dir_parametric_k2.json",
    "heads_dir_parametric_k3.json",
    "heads_dir_parametric_k2_clean.json",
    "split.json",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

for rel in COMMON_FILES:
    src = ROOT / rel
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

data_src = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
data_dst = OUT / "data" / "conflictqa_pilot250_popular.jsonl"
data_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(data_src, data_dst)

heads_out = OUT / "directional_heads"
heads_out.mkdir(parents=True, exist_ok=True)
for name in HEAD_FILES:
    shutil.copy2(HEADS_DIR / name, heads_out / name)

nb_src = ROOT / "juice_repro" / "kaggle_stage2_base_cell3_directional.ipynb"
nb_dst = OUT / "juice_repro" / "kaggle_stage2_base_cell3_directional.ipynb"
nb_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, nb_dst)

if ZIP.exists():
    ZIP.unlink()

with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            arc = path.relative_to(OUT).as_posix()
            zf.write(path, arc)

print("Wrote", OUT)
print("Wrote", ZIP)
print("Files:")
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT))
