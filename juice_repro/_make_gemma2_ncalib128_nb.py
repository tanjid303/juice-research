"""Generate juice_repro/kaggle_gemma2_ncalib128_quantize.ipynb"""
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
    """# Gemma-2-9B GPTQ-INT4 — N_CALIB=128 (quantize only)

Canonical INT4 for the Gemma-2 Stage 2 copy of the Llama protocol.

**New notebook.** Do **not** re-run the smoke notebook. No FP16 JuICE wrap in this job.

Same locked recipe as Llama: gptqmodel 2.2, transformers 4.51.3, 4-bit,
`group_size=128`, `sym=True`, `desc_act=True`, C4 × 128 truncated to 512 tokens.
Smoke load is **`BACKEND.TORCH`** (Triton packing breaks `wf_unsqueeze_zero`).
Decode **strings**, not token tensors.

Output: `/kaggle/working/gemma2_9b_base_gptq_int4_ncalib128/`

**~60–120 min** on GPU T4 ×2.

## Before you start

1. **GPU T4 ×2**, Internet **ON**, secret `HF_TOKEN`
2. Accept [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)
3. Attach **`kaggle_juice_gemma2_ncalib128.zip`**
4. **Run All in the draft editor.** Do **not** click Save Version first.
   A failed Save Version wipes `/kaggle/working` (that is Version 2).
5. When it prints `SAVE VERSION NOW`, use **Quick Save** (not Save & Run All).
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
    assert hits, f"Missing {name} — attach kaggle_juice_gemma2_ncalib128.zip"
    hits.sort(key=lambda p: (0 if "ncalib128" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT_01 = find_one("gemma2_ncalib128_quantize.py")
print("SCRIPT_01", SCRIPT_01)
"""
)

md("## 1 — Quantize (N_CALIB=128)")
code(
    r"""print("=== STAGE 1: Gemma-2-9B N_CALIB=128 ===")
exec(compile(SCRIPT_01.read_text(encoding="utf-8"), str(SCRIPT_01), "exec"))
print("=== STAGE 1 done ===")
"""
)

md("## 2 — Confirm weights, then Save Version")
code(
    r"""from pathlib import Path
out = Path("/kaggle/working/gemma2_9b_base_gptq_int4_ncalib128")
print(out, "exists" if out.is_dir() else "MISSING")
assert out.is_dir() and (out / "config.json").is_file()
weights = [p for p in out.iterdir() if p.suffix in {".safetensors", ".bin"}]
print("weights:")
for p in weights:
    print(f"  {p.name:40s} {p.stat().st_size/1024**2:.1f} MiB")
assert weights, "No .safetensors — quantize failed"
for name in ("quant_meta.json", "decode_smoke.json"):
    p = out / name
    print(name, "ok" if p.is_file() else "MISSING")
    if p.is_file():
        print(p.read_text(encoding="utf-8")[:2000])
print()
print("=" * 60)
print("SAVE VERSION NOW — use Quick Save, not Save & Run All")
print("Attach this notebook output to later Gemma Stage 2 notebooks:")
print("  Add Input → Notebook Output Files → this version")
print("  folder:", out)
print("Do not use gemma2_gptq_ncalib8.")
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
out = Path(__file__).resolve().parent / "kaggle_gemma2_ncalib128_quantize.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("Wrote", out)
