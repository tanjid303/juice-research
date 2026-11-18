"""Kaggle zip: Gemma-2-9B base FP16 Cell-3 apply (no GPTQ)."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma2_base_cell3"
ZIP = ROOT / "kaggle_juice_gemma2_base_cell3.zip"

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

HEADS_DIR = ROOT / "juice_stage2_gemma2_base_id" / "directional_heads"
HEAD_FILES = [
    "heads_dir_context_k10.json",
    "heads_dir_parametric_k2.json",
    "heads_dir_parametric_k2_clean.json",
    "heads_dir_parametric_k1_raw.json",
    "split.json",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_gemma2_base_cell3_nb.py")]
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
for name in HEAD_FILES:
    shutil.copy2(HEADS_DIR / name, heads / name)

nb = ROOT / "juice_repro" / "kaggle_gemma2_base_cell3.ipynb"
shutil.copy2(nb, OUT / "kaggle_gemma2_base_cell3.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb, OUT / "juice_repro" / "kaggle_gemma2_base_cell3.ipynb")

readme = """# Gemma-2-9B base Cell-3 FP16 apply

Does **not** use the INT4 checkpoint. Same 50 eval IDs as Gemma ID / Llama Stage 2.

| Condition | Heads |
|-----------|-------|
| steer_context_k10 | directional ctx top-10 (headline) |
| steer_parametric_k2_clean | L31_H7, L39_H6 |
| steer_parametric_k1_raw | L27_H6 (Llama L1_H4 analog) |
| random_context_k10 | seed 42, exclude identified set |

Protocol: `juice_coherent`, greedy, α+=1, α−=0, paired bootstrap 1000 / seed 42.

## Run

1. New notebook, GPU **T4 x2**, Internet ON, secret `HF_TOKEN`
2. Upload `kaggle_juice_gemma2_base_cell3.zip` as a dataset
3. File → Upload `kaggle_gemma2_base_cell3.ipynb`
4. Run All in the **draft** (~1.5–3 h). Do **not** Save Version first.
5. Quick Save when it prints `SAVE VERSION NOW`.

Need: `juice_stage2_gemma2_base_cell3/gemma2_base_cell3_summary.json` and `gemma2_base_cell3_bootstrap.json`.

Do **not** start GPTQ apply/transfer until the FP16 gate table is reported.
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
