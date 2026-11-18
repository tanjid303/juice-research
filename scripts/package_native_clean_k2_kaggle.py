"""Assemble a Kaggle zip that only runs the native INT4 clean K2 transfer cell.

Weights are NOT in this zip. Attach the N_CALIB=128 checkpoint as notebook output.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_int4_native_clean_k2"
ZIP = ROOT / "kaggle_juice_stage2_int4_native_clean_k2.zip"

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
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_native_clean_k2_nb.py")]
)

for rel in COMMON_FILES + ["scripts/ncalib128_04_native_clean_k2.py"]:
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
shutil.copy2(
    ROOT / "juice_stage2_int4_native_id" / "directional_heads" / "heads_int4_native_parametric_k2_clean.json",
    heads_out / "heads_int4_native_parametric_k2_clean.json",
)
shutil.copy2(
    ROOT / "juice_stage2_base_id" / "directional_heads" / "split.json",
    heads_out / "split.json",
)

ref_src = ROOT / "this_is_the_128" / "steer_baseline (2).json"
if ref_src.is_file():
    ref_dst = OUT / "ncalib128_reference"
    ref_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref_src, ref_dst / "steer_baseline.json")

nb_src = ROOT / "juice_repro" / "kaggle_stage2_int4_native_clean_k2.ipynb"
# Zip root + juice_repro/ so File → Upload notebook is one click.
shutil.copy2(nb_src, OUT / "kaggle_stage2_int4_native_clean_k2.ipynb")
(OUT / "juice_repro").mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, OUT / "juice_repro" / "kaggle_stage2_int4_native_clean_k2.ipynb")

readme = """# Kaggle — native INT4 clean K2 on N_CALIB=128

Steer `L16_H30` + `L17_H27` (native INT4 clean-filter pair, never steered) on the
existing `llama31_8b_base_gptq_int4_ncalib128` checkpoint.

**This zip has no weights.** Attach **ungabunga** output **346038437**.

## 1. Upload this zip as a dataset

File: `kaggle_juice_stage2_int4_native_clean_k2.zip`

## 2. Attach the N_CALIB=128 checkpoint (required)

In the notebook sidebar:

1. **Add Input → Notebook Output Files** (not Datasets)
2. Pick notebook **ungabunga**, version **346038437**
3. Confirm you see `llama31_8b_base_gptq_int4_ncalib128/` with `config.json` and `*.safetensors`

Do **not** attach `llama31_8b_base_gptq_int4` (that is N_CALIB=64).

## 3. Upload the notebook

**File → Upload notebook** → `kaggle_stage2_int4_native_clean_k2.ipynb` (zip root).

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Datasets | this zip **and** the N_CALIB=128 checkpoint |
| Secret | `HF_TOKEN` only if tokenizer files are missing |

## 4. Save & Run All

~1.5–3 h. Download:

```
/kaggle/working/juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json
/kaggle/working/juice_stage2_int4_ncalib128/int4_native_clean_k2_bootstrap.json
/kaggle/working/juice_stage2_int4_ncalib128/steer_int4_native_parametric_k2_clean.json
```

Locked: K=2, α+=1, α−=0, 50 ids in `directional_heads/split.json` (seed 42),
prompt `juice_coherent`, greedy 16 tokens, bootstrap 1000 / seed 42.
JuICE dual-run port (`common/juice.py`, `common/juice_attn.py`) is copied unchanged.
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
print("\nContents:")
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(OUT), path.stat().st_size)
