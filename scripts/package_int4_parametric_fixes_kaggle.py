"""Assemble Kaggle zip for INT4 parametric recovery checks."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_int4_parametric_fixes"
ZIP = ROOT / "kaggle_juice_stage2_int4_parametric_fixes.zip"

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
    "heads_dir_parametric_k2_clean.json",
    "split.json",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_int4_parametric_fixes_nb.py")]
)

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

# INT4 ID score maps for check 2 (no GPU)
shutil.copy2(
    ROOT / "juice_stage2_int4_native_id" / "id_diagnostics_raw.json",
    OUT / "id_diagnostics_raw.json",
)

nb_src = ROOT / "juice_repro" / "kaggle_stage2_int4_parametric_fixes.ipynb"
nb_dst = OUT / "juice_repro" / "kaggle_stage2_int4_parametric_fixes.ipynb"
nb_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, nb_dst)

(OUT / "KAGGLE.md").write_text(
    """# Kaggle — INT4 parametric recovery (L1_H4 + alpha grid + suppress ctx K10)

## Attach
1. This zip as a dataset
2. `quantizing-output` (GPTQ weights)

## Notebook
Upload `juice_repro/kaggle_stage2_int4_parametric_fixes.ipynb`

GPU T4 x2, Internet ON. ~40–70 min. **Do not** re-run native ID.

GPU order after baseline: L1_H4 (K=1) → clean-pair alpha grid → suppress ctx K10.

Output: `/kaggle/working/juice_stage2_int4_parametric_fixes/int4_parametric_fixes_summary.json`
Look at `l1_h4` first: negative delta with CI excluding 0 would mean parametric steering
works on INT4 via the raw-ID peak (L1_H4), not the directional clean pair.
""",
    encoding="utf-8",
)

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(OUT).as_posix())

print("Wrote", ZIP, "size", ZIP.stat().st_size)
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT))
