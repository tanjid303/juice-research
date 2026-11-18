"""Package Kaggle upload zip for base-model GPTQ quantization notebook."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_quantize_base"
ZIP = ROOT / "kaggle_quantize_llama31_8b_base.zip"

COMMON_FILES = [
    "common/__init__.py",
    "common/models.py",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# Regenerate notebook
import subprocess

subprocess.check_call(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "juice_repro" / "_make_quantize_base_nb.py")]
)

nb_src = ROOT / "juice_repro" / "kaggle_quantize_llama31_8b_base_gptq.ipynb"
nb_dst = OUT / "juice_repro" / "kaggle_quantize_llama31_8b_base_gptq.ipynb"
nb_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(nb_src, nb_dst)

for rel in COMMON_FILES:
    src = ROOT / rel
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

(OUT / "KAGGLE.md").write_text(
    """# Kaggle — Quantize Llama-3.1-8B base → GPTQ INT4

## 1. Upload dataset
- Upload **`kaggle_quantize_llama31_8b_base.zip`** as a new Kaggle Dataset.

## 2. New notebook
- **File → New Notebook → Upload Notebook**
- Pick `juice_repro/kaggle_quantize_llama31_8b_base_gptq.ipynb` from the dataset  
  OR copy notebook from dataset into a fresh notebook (delete old auto-gptq notebook).

## 3. Notebook settings
| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Secret | `HF_TOKEN` |

## 4. Hugging Face
Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B

## 5. Run
- **Save & Run All** (not Run cell-by-cell on a stale kernel)
- First run: `N_CALIB = 64` in cell 2 (smoke, ~20–30 min)
- Then: `N_CALIB = 128`, then `512` for final checkpoint
- **Do not skip truncation** — cell 4 tokenizes with `MAX_SEQ_LEN = 512` (full C4 docs OOM)

## 6. Output
```
/kaggle/working/llama31_8b_base_gptq_int4/
```
**Save Version** → download → re-upload as dataset for INT4 JuICE transfer.

## Tooling
- **`gptqmodel==2.2.0`** + **`transformers==4.51.3`** (not 4.56+/5.x — those break RoPE on layer 1)
- Does **NOT** use auto-gptq (no Py3.12 wheel)
- **Restart kernel** after upload; do not reuse a mid-quantize crash

## Plan B
Community checkpoint: `ModelCloud/Meta-Llama-3.1-8B-gptq-4bit`
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
