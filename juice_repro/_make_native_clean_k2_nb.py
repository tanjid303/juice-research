"""Generate juice_repro/kaggle_stage2_int4_native_clean_k2.ipynb"""
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
    """# Stage 2 — native INT4 clean K2 (`L16_H30`, `L17_H27`) on N_CALIB=128

Steer the native INT4 clean-filter pair that was identified but never applied.

| Locked | Value |
|--------|--------|
| Checkpoint | **ungabunga** output `346038437` → `llama31_8b_base_gptq_int4_ncalib128/` |
| Heads | `L16_H30`, `L17_H27` — K=2, parametric, α+=1, α−=0 |
| Eval | same 50 ids (`directional_heads/split.json`, seed 42) |
| Prompt | `juice_coherent`, greedy, 16 new tokens |
| Stats | paired bootstrap 1000 draws, seed 42 |
| Runtime | **~1.5–3 h** on GPU T4 ×2 |

These heads have Jaccard 0.00 vs the FP16 clean pair (`L20_H14`/`L26_H15`). There is **no FP16 same-sign gate**. A usable parametric steer needs Δ negative, CI excluding 0, and `other` not exploding.

## Before Run All

1. **GPU T4 ×2**, Internet **ON**
2. Attach **this zip** as a dataset (`kaggle_juice_stage2_int4_native_clean_k2.zip`)
3. **Add Input → Notebook Output Files** → notebook **ungabunga**, version **346038437**. Confirm the sidebar shows `llama31_8b_base_gptq_int4_ncalib128/` with `config.json` and `*.safetensors`. The code zip alone is not enough.
4. **File → Upload notebook** `kaggle_stage2_int4_native_clean_k2.ipynb` from this zip (also at `juice_repro/` inside the zip)
5. Save & Run All (fresh kernel)

Do **not** attach the old N=64 folder `llama31_8b_base_gptq_int4`.
"""
)

md("## 0 — Locate zip + confirm checkpoint")
code(
    r"""import os, json
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
    assert hits, f"Missing {name} — attach kaggle_juice_stage2_int4_native_clean_k2.zip"
    hits.sort(key=lambda p: (0 if "native_clean" in str(p).lower() or "ncalib128" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT = find_one("ncalib128_04_native_clean_k2.py")
HEADS = find_one("heads_int4_native_parametric_k2_clean.json")
SPLIT = find_one("split.json")
REPO = SCRIPT.parent.parent
print("REPO", REPO)
print("HEADS", HEADS)
print("SPLIT", SPLIT)

print("\n=== /kaggle/input (top two levels) ===")
inp = Path("/kaggle/input")
if not inp.exists():
    print("  MISSING")
else:
    for top in sorted(inp.iterdir()):
        print(f"  {'DIR ' if top.is_dir() else 'FILE'} {top}")
        if top.is_dir():
            kids = sorted(top.iterdir())
            for c in kids[:40]:
                print(f"    {'DIR ' if c.is_dir() else 'FILE'} {c.name}")
            if len(kids) > 40:
                print(f"    ... {len(kids) - 40} more")

def looks_gptq(p: Path) -> bool:
    if not p.is_dir() or not (p / "config.json").is_file():
        return False
    names = {x.name.lower() for x in p.iterdir() if x.is_file()}
    if not any(n.endswith(".safetensors") or n.endswith(".bin") for n in names):
        return False
    if "quantize_config.json" in names or "quant_meta.json" in names:
        return True
    cfg = (p / "config.json").read_text(encoding="utf-8", errors="ignore").lower()
    return "gptq" in cfg or "quantization_config" in cfg

def n_calib_of(p: Path):
    meta = p / "quant_meta.json"
    if not meta.is_file():
        return None
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("n_calib"))
    except Exception:
        return None

print("\n=== GPTQ dirs (config.json + weights) ===")
gptq = []
for root in (Path("/kaggle/working"), Path("/kaggle/input")):
    if not root.exists():
        continue
    for cfg in root.rglob("config.json"):
        d = cfg.parent
        if looks_gptq(d) and d not in gptq:
            gptq.append(d)
            print(f"  n_calib={n_calib_of(d)}  {d}")

# Pin to ungabunga if that mount exists. Uncomment/edit only if cell 0 prints a different path.
UNGABUNGA_DEFAULTS = [
    Path("/kaggle/input/ungabunga/llama31_8b_base_gptq_int4_ncalib128"),
    Path("/kaggle/input/ungabunga/working/llama31_8b_base_gptq_int4_ncalib128"),
]
for p in UNGABUNGA_DEFAULTS:
    if looks_gptq(p) and p not in gptq:
        gptq.append(p)
        print(f"  n_calib={n_calib_of(p)}  {p}")

ranked = []
for d in gptq:
    s = str(d).replace("\\", "/").lower()
    n = n_calib_of(d)
    score = (
        (200 if "ungabunga" in s else 0)
        + (100 if "ncalib128" in s else 0)
        + (50 if n == 128 else 0)
    )
    if d.name == "llama31_8b_base_gptq_int4" and "ncalib128" not in s and n != 128:
        print("  skip N=64", d)
        continue
    if score > 0:
        ranked.append((score, d))
ranked.sort(reverse=True)
if "INT4_MODEL_ID_OVERRIDE" in globals() and INT4_MODEL_ID_OVERRIDE:
    CKPT = Path(INT4_MODEL_ID_OVERRIDE)
    print("using INT4_MODEL_ID_OVERRIDE", CKPT)
elif ranked:
    CKPT = ranked[0][1]
    INT4_MODEL_ID_OVERRIDE = str(CKPT)
    print("auto-picked N=128 checkpoint:", CKPT)
else:
    n64 = [d for d in gptq if d.name == "llama31_8b_base_gptq_int4"]
    msg = (
        "N_CALIB=128 weights are not in this kernel.\n"
        "Add Input → Notebook Output Files → ungabunga version 346038437 "
        "(folder llama31_8b_base_gptq_int4_ncalib128 with config.json + *.safetensors).\n"
        "Do not attach only this code zip, and do not attach llama31_8b_base_gptq_int4 (N=64)."
    )
    if n64:
        msg += f"\nCurrently attached N=64 checkpoint: {n64[0]}"
    raise FileNotFoundError(msg)
print("INT4_MODEL_ID_OVERRIDE", INT4_MODEL_ID_OVERRIDE)
"""
)

md(
    """## 1 — Run the transfer cell

Uses `common/juice.py` + `common/juice_attn.py` dual-run port unchanged.
Writes `/kaggle/working/juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json`.
"""
)
code(
    r"""print("=== native INT4 clean K2 (L16_H30, L17_H27) on N_CALIB=128 ===")
exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"))
print("=== done ===")
"""
)

md("## 2 — Outputs to download")
code(
    r"""from pathlib import Path
out = Path("/kaggle/working/juice_stage2_int4_ncalib128")
print(out)
for p in sorted(out.glob("int4_native_clean_k2*.json")) + sorted(out.glob("steer_int4_native*.json")) + sorted(out.glob("steer_baseline_native*.json")):
    print(f"  {p.name:50s} {p.stat().st_size/1024:.1f} KiB")
print()
print("Save Version → download these three:")
print("  juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json")
print("  juice_stage2_int4_ncalib128/int4_native_clean_k2_bootstrap.json")
print("  juice_stage2_int4_ncalib128/steer_int4_native_parametric_k2_clean.json")
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
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_native_clean_k2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
