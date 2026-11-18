"""Kaggle zip: Cell A α-sweep. FP16 only — no INT4 weights."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_cell_a_alpha_sweep"
ZIP = ROOT / "kaggle_juice_stage2_cell_a_alpha_sweep.zip"

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
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_cell_a_alpha_sweep_nb.py")]
)
for rel in COMMON + ["scripts/ncalib128_06_cell_a_alpha_sweep.py"]:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)

(OUT / "data").mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "data" / "conflictqa_pilot250_popular.jsonl", OUT / "data" / "conflictqa_pilot250_popular.jsonl")

heads = OUT / "directional_heads"
heads.mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "juice_stage2_int4_native_id" / "directional_heads" / "heads_int4_native_context_k10.json",
    heads / "heads_int4_native_context_k10.json",
)
shutil.copy2(ROOT / "juice_stage2_base_id" / "directional_heads" / "split.json", heads / "split.json")

fp16 = OUT / "fp16_reference"
fp16.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "juice_stage2_base_directional_cell3" / "steer_baseline.json", fp16 / "steer_baseline.json")

# Reuse α=1 from the reverse-transfer run.
a1 = ROOT / "unga-bunga-23" / "juice_stage2_int4_ncalib128" / "steer_reverse_int4_context_k10_on_fp16.json"
if a1.is_file():
    shutil.copy2(a1, OUT / "steer_reverse_int4_context_k10_on_fp16.json")

nb = ROOT / "juice_repro" / "kaggle_stage2_cell_a_alpha_sweep.ipynb"
shutil.copy2(nb, OUT / "kaggle_stage2_cell_a_alpha_sweep.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_stage2_cell_a_alpha_sweep.ipynb")

readme = """# Cell A α-sweep (INT4 K10 → FP16)

No INT4 weights. α+ = 0.25, 0.5, 1.0. α=1 reused from bundled reverse-transfer rows.

1. Upload `kaggle_juice_stage2_cell_a_alpha_sweep.zip` as a dataset
2. File → Upload notebook `kaggle_stage2_cell_a_alpha_sweep.ipynb`
3. GPU T4 x2, Internet ON, HF_TOKEN
4. Save & Run All (~2–3 h)

Download `cell_a_alpha_sweep_summary.json`.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
(OUT / "KAGGLE.md").write_text(readme, encoding="utf-8")

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(OUT).as_posix())
print("Wrote", ZIP)
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT), path.stat().st_size)
