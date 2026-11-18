"""Generate juice_repro/kaggle_stage2_int4_ncalib128.ipynb"""
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
    """# Stage 2 QC — re-quantize Llama-3.1-8B base at N_CALIB=128, re-run 3 cells

One Kaggle session (GPU T4 ×2, 12 h). **Only** `N_CALIB` and the checkpoint path differ from the N=64 smoke recipe.

| Script | What | Expected runtime |
|--------|------|------------------|
| `scripts/ncalib128_01_quantize.py` | GPTQ-INT4, N_CALIB=128, save to `llama31_8b_base_gptq_int4_ncalib128/` (does **not** overwrite N=64) | **~45–90 min** |
| `scripts/ncalib128_02_rerun_three_cells.py` | baseline + context K10 + random-K10 (seed 42) + parametric clean K2 (L20_H14, L26_H15); paired bootstrap 1000 / seed 42 | **~3–6 h** |
| `scripts/ncalib128_03_compare.py` | N64 vs N128 table (same sign? excludes-0 flip?) | **<1 min** |
| `scripts/ncalib128_04_native_clean_k2.py` | native INT4 clean pair L16_H30 + L17_H27, K=2, α+=1 (never steered before) | **~1.5–3 h** |

**Total ~5.5–11 h.** Fits the 12 h limit, but see the note before cell 4 if stage 2 ran long.

## Before Run All

1. **GPU T4 ×2**, Internet ON, `HF_TOKEN` secret
2. Accept [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) license
3. Attach **this zip** as a dataset
4. **Do not** attach the old N=64 `llama31_8b_base_gptq_int4` as the model to apply — script 01 writes a new folder; script 02 loads only `*ncalib128*`
5. Save & Run All (fresh kernel)
"""
)

md("## 0 — Locate dataset scripts")
code(
    r"""import os, sys
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
    assert hits, f"Missing {name} — attach kaggle_juice_stage2_int4_ncalib128.zip"
    hits.sort(key=lambda p: (0 if "ncalib128" in str(p) else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT_01 = find_one("ncalib128_01_quantize.py")
SCRIPT_02 = find_one("ncalib128_02_rerun_three_cells.py")
SCRIPT_03 = find_one("ncalib128_03_compare.py")
SCRIPT_04 = find_one("ncalib128_04_native_clean_k2.py")
REPO = SCRIPT_01.parent.parent
print("REPO", REPO)

import shutil
n64_src = REPO / "n64_baseline"
n64_dst = Path("/kaggle/working/n64_baseline")
if n64_src.is_dir() and not n64_dst.exists():
    shutil.copytree(n64_src, n64_dst)
    print("copied n64_baseline ->", n64_dst)
"""
)

md(
    """## 1 — Re-quantize (N_CALIB=128)

Same recipe as `juice_repro/_make_quantize_base_nb.py`: gptqmodel 2.2, transformers 4.51.3,
4-bit, `group_size=128`, `sym=True`, `desc_act=True`, C4 truncated to 512 tokens.
Output: `/kaggle/working/llama31_8b_base_gptq_int4_ncalib128/`
"""
)
code(
    r"""print("=== STAGE 1: quantize N_CALIB=128 ===")
exec(compile(SCRIPT_01.read_text(encoding="utf-8"), str(SCRIPT_01), "exec"))
print("=== STAGE 1 done ===")
"""
)

md(
    """## 2 — Re-run three cells on the new checkpoint

Uses the existing JuICE dual-run port (`common/juice.py` / `common/juice_attn.py`) unchanged.
Same 50 `eval_ids` from `directional_heads/split.json` (seed 42). Same locked FP16 heads.
Random-K10: `random_head_control(..., seed=42, exclude=FP16 context K10)`.
Bootstrap: `bootstrap_paired_follow_delta(..., n_boot=1000, seed=42)`.
"""
)
code(
    r"""print("=== STAGE 2: three cells vs N_CALIB=128 ===")
exec(compile(SCRIPT_02.read_text(encoding="utf-8"), str(SCRIPT_02), "exec"))
print("=== STAGE 2 done ===")
"""
)

md(
    """## 3 — N64 vs N128 comparison

Loads bundled `n64_baseline/` (the original smoke-calib results) next to the new JSON.
"""
)
code(
    r"""print("=== STAGE 3: N64 vs N128 table ===")
exec(compile(SCRIPT_03.read_text(encoding="utf-8"), str(SCRIPT_03), "exec"))
print("=== STAGE 3 done ===")
"""
)

md(
    """## 4 — Native INT4 clean pair L16_H30 + L17_H27 (never steered)

Resolves the standing flag "native INT4 clean pair was never steered." Same checkpoint,
same 50 items, K=2, α+=1, α−=0, bootstrap 1000 / seed 42.

These heads have **Jaccard 0.00** vs the FP16 clean pair, so there is no FP16 same-sign
gate — read it on its own terms: a usable parametric steer needs Δ negative with CI
excluding 0 and `other` not exploding.

**If the session is already past ~9 h, stop here** and run this cell in a second session
(attach the checkpoint from this run's Save Version); it needs ~1.5–3 h.
"""
)
code(
    r"""print("=== STAGE 4: native INT4 clean K2 (L16_H30, L17_H27) ===")
exec(compile(SCRIPT_04.read_text(encoding="utf-8"), str(SCRIPT_04), "exec"))
print("=== STAGE 4 done ===")
"""
)

md("## Outputs")
code(
    r"""from pathlib import Path
print("Checkpoint:", Path("/kaggle/working/llama31_8b_base_gptq_int4_ncalib128"))
print("Results:   ", Path("/kaggle/working/juice_stage2_int4_ncalib128"))
for p in sorted(Path("/kaggle/working/juice_stage2_int4_ncalib128").glob("*.json")):
    print(f"  {p.name:40s} {p.stat().st_size/1024:.1f} KiB")
print("Save Version → download:")
print("  llama31_8b_base_gptq_int4_ncalib128/")
print("  juice_stage2_int4_ncalib128/int4_transfer_bootstrap.json")
print("  juice_stage2_int4_ncalib128/n64_vs_n128_comparison.json")
print("  juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json")
print("  juice_stage2_int4_ncalib128/int4_native_clean_k2_bootstrap.json")
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
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_ncalib128.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
