"""Assemble kaggle upload folder + zip for FP16→INT4 JuICE transfer."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_int4_transfer"
ZIP = ROOT / "kaggle_juice_stage2_int4_transfer.zip"

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

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_int4_transfer_nb.py")]
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

nb_src = ROOT / "juice_repro" / "kaggle_stage2_int4_transfer.ipynb"
nb_dst = OUT / "juice_repro" / "kaggle_stage2_int4_transfer.ipynb"
nb_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, nb_dst)

(OUT / "KAGGLE.md").write_text(
    """# Kaggle — Stage 2 FP16→INT4 transfer (base)

Weights are **not** in this zip. Attach the quantized model separately.

## 1. Upload this zip as a dataset
`kaggle_juice_stage2_int4_transfer.zip`

## 2. Attach the GPTQ checkpoint (required — this zip has no weights)

In the transfer notebook sidebar:

1. **Add Input → Notebook Output Files** (not “Datasets”)
2. Pick the **quantize** notebook’s **Save Version** (the run that printed `Done. Download: /kaggle/working/llama31_8b_base_gptq_int4`)
3. Confirm you see a folder with `config.json` and `*.safetensors` (~4–6 GB)

The code zip (`1234567890` / `kaggle_juice_stage2_int4_transfer`) is not the model.

If auto-detect still fails, set in cell 2:

```python
INT4_MODEL_ID_OVERRIDE = "/kaggle/input/<slug>/llama31_8b_base_gptq_int4"
```

## 3. New notebook
Upload `juice_repro/kaggle_stage2_int4_transfer.ipynb`.

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Datasets | this zip **and** the GPTQ checkpoint |
| Secret | `HF_TOKEN` only if tokenizer files are missing |

## 4. Run
Save & Run All. Output:

```
/kaggle/working/juice_stage2_int4_transfer/int4_transfer_summary.json
```

Gate: context K10/K5 and parametric clean K2 must keep the **same sign** as FP16
(+10 / +6 / −8 pp) with coherence ≥ 0.90.
""",
    encoding="utf-8",
)

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
        print(" ", path.relative_to(OUT))
