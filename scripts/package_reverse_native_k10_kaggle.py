"""Kaggle zip: reverse transfer + native INT4 context K10. No weights."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_int4_reverse_native_k10"
ZIP = ROOT / "kaggle_juice_stage2_int4_reverse_native_k10.zip"

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

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

subprocess.check_call(
    [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "juice_repro" / "_make_reverse_native_k10_nb.py"),
    ]
)

for rel in COMMON_FILES + ["scripts/ncalib128_05_reverse_and_native_k10.py"]:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)

data_dst = OUT / "data" / "conflictqa_pilot250_popular.jsonl"
data_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "data" / "conflictqa_pilot250_popular.jsonl", data_dst)

heads_out = OUT / "directional_heads"
heads_out.mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "juice_stage2_int4_native_id" / "directional_heads" / "heads_int4_native_context_k10.json",
    heads_out / "heads_int4_native_context_k10.json",
)
shutil.copy2(
    ROOT / "juice_stage2_base_id" / "directional_heads" / "heads_dir_context_k10.json",
    heads_out / "heads_dir_context_k10.json",
)
shutil.copy2(
    ROOT / "juice_stage2_base_id" / "directional_heads" / "split.json",
    heads_out / "split.json",
)

fp16_ref = OUT / "fp16_reference"
fp16_ref.mkdir(parents=True, exist_ok=True)
shutil.copy2(
    ROOT / "juice_stage2_base_directional_cell3" / "steer_baseline.json",
    fp16_ref / "steer_baseline.json",
)

int4_ref = OUT / "ncalib128_reference"
int4_ref.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "this_is_the_128" / "steer_baseline (2).json", int4_ref / "steer_baseline.json")

nb_src = ROOT / "juice_repro" / "kaggle_stage2_int4_reverse_native_k10.ipynb"
shutil.copy2(nb_src, OUT / "kaggle_stage2_int4_reverse_native_k10.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, OUT / "juice_repro" / "kaggle_stage2_int4_reverse_native_k10.ipynb")

readme = """# Kaggle — reverse transfer + native INT4 context K10

Cell A: native INT4 context K10 heads → FP16.
Cell B: same heads → N_CALIB=128 INT4 (in-precision).

**This zip has no weights.** Attach **ungabunga** output **346038437**.

## Run

1. Upload `kaggle_juice_stage2_int4_reverse_native_k10.zip` as a dataset
2. **File → Upload notebook** `kaggle_stage2_int4_reverse_native_k10.ipynb`
3. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
4. **Add Input → Notebook Output Files** → **ungabunga** version **346038437**
5. Save & Run All (~3–5 h)

Download:

```
/kaggle/working/juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_summary.json
/kaggle/working/juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_bootstrap.json
```

Locked: INT4 heads from `heads_int4_native_context_k10.json` (L7_H26 … L9_H0).
FP16 K10 (comparison only) from `heads_dir_context_k10.json`. Jaccard 0.33.
Baselines reused; α+=1, α−=0, split.json seed 42, bootstrap 1000 / seed 42.
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
