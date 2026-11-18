"""Generate juice_repro/kaggle_stage2_int4_ncalib128_quantize_only.ipynb

Rebuild the lost N_CALIB=128 checkpoint. Does not run transfer cells.
"""
import json
from pathlib import Path

cells = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.split("\n")]})


def code(s: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [line + "\n" for line in s.split("\n")],
        }
    )


md(
    """# Rebuild N_CALIB=128 GPTQ checkpoint (quantize only)

This notebook **only** re-quantizes Llama-3.1-8B base → GPTQ-INT4 with `N_CALIB=128`.
It does **not** re-run transfer cells.

Same locked recipe as the lost run: gptqmodel 2.2, transformers 4.51.3, 4-bit,
`group_size=128`, `sym=True`, `desc_act=True`, C4 × 128 truncated to 512 tokens.
Output: `/kaggle/working/llama31_8b_base_gptq_int4_ncalib128/`

**~45–90 min** on GPU T4 ×2.

## Before Run All

1. **GPU T4 ×2**, Internet **ON**, secret `HF_TOKEN`
2. Accept [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) license
3. Attach **this zip** as a dataset
4. Save & Run All
5. When it prints `SAVE VERSION NOW`, click **Save Version** immediately. That output is the checkpoint you attach to the native-clean-K2 notebook.
"""
)

md("## 0 — Locate script")
code(
    r"""import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name} — attach kaggle_juice_stage2_int4_ncalib128_quantize_only.zip"
    hits.sort(key=lambda p: (0 if "ncalib128" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT_01 = find_one("ncalib128_01_quantize.py")
print("SCRIPT_01", SCRIPT_01)
"""
)

md("## 1 — Quantize (N_CALIB=128)")
code(
    r"""print("=== STAGE 1: quantize N_CALIB=128 ===")
exec(compile(SCRIPT_01.read_text(encoding="utf-8"), str(SCRIPT_01), "exec"))
print("=== STAGE 1 done ===")
"""
)

md("## 2 — Confirm weights, then Save Version")
code(
    r"""from pathlib import Path
out = Path("/kaggle/working/llama31_8b_base_gptq_int4_ncalib128")
print(out, "exists" if out.is_dir() else "MISSING")
assert out.is_dir() and (out / "config.json").is_file()
weights = [p for p in out.iterdir() if p.suffix in {".safetensors", ".bin"}]
print("weights:")
for p in weights:
    print(f"  {p.name:40s} {p.stat().st_size/1024**2:.1f} MiB")
assert weights, "No .safetensors — quantize failed"
meta = out / "quant_meta.json"
if meta.is_file():
    print(meta.read_text(encoding="utf-8")[:800])
print()
print("=" * 60)
print("SAVE VERSION NOW")
print("Then attach this notebook's output to the native-clean-K2 notebook:")
print("  Add Input → Notebook Output Files → this Save Version")
print("  folder:", out)
print("=" * 60)
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "gpu",
    },
    "cells": cells,
}
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_ncalib128_quantize_only.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
