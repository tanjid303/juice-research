"""Generate juice_repro/kaggle_stage2_int4_reverse_native_k10.ipynb"""
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
    """# Stage 2 — reverse transfer + native INT4 context K10 (N_CALIB=128)

Two sanity-check cells on the locked dual-run port. Same 50 eval ids, α+=1, α−=0.

| Cell | Identify on | Apply to | Comparator |
|------|-------------|----------|------------|
| **A** reverse | native INT4 context K10 | FP16 `meta-llama/Llama-3.1-8B` | FP16-ID K10 on FP16 = **+10 pp** |
| **B** native | native INT4 context K10 | `llama31_8b_base_gptq_int4_ncalib128` | FP16-ID K10 transferred onto INT4 = **+18 pp** |

Head sets are **not** the same (Jaccard 0.33, 5/15 overlap). Both are listed in the output JSON.

Baselines are **reused** (not recomputed): FP16 Cell-3 `steer_baseline.json` (fc 16%) and N=128 `steer_baseline.json` (fc 20%).

Runtime **~3–5 h** on GPU T4 ×2 (INT4 steer, then FP16 steer).

## Before Run All

1. **GPU T4 ×2**, Internet **ON**, secret `HF_TOKEN`
2. Attach **this zip** as a dataset
3. **Add Input → Notebook Output Files** → **ungabunga** version **346038437** (`llama31_8b_base_gptq_int4_ncalib128/`)
4. Accept the Llama-3.1-8B license
5. **File → Upload notebook** `kaggle_stage2_int4_reverse_native_k10.ipynb`
6. Save & Run All
"""
)

md("## 0 — Locate zip + ungabunga checkpoint")
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
    assert hits, f"Missing {name} — attach kaggle_juice_stage2_int4_reverse_native_k10.zip"
    hits.sort(key=lambda p: (0 if "reverse" in str(p).lower() or "ncalib" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT = find_one("ncalib128_05_reverse_and_native_k10.py")
print("SCRIPT", SCRIPT)

print("\n=== /kaggle/input (top two levels) ===")
inp = Path("/kaggle/input")
if inp.exists():
    for top in sorted(inp.iterdir()):
        print(f"  {'DIR ' if top.is_dir() else 'FILE'} {top}")
        if top.is_dir():
            for c in sorted(top.iterdir())[:30]:
                print(f"    {'DIR ' if c.is_dir() else 'FILE'} {c.name}")

def looks_gptq(p: Path) -> bool:
    if not p.is_dir() or not (p / "config.json").is_file():
        return False
    names = {x.name.lower() for x in p.iterdir() if x.is_file()}
    return any(n.endswith(".safetensors") or n.endswith(".bin") for n in names)

gptq = []
for root in (Path("/kaggle/working"), Path("/kaggle/input")):
    if not root.exists():
        continue
    for cfg in root.rglob("config.json"):
        d = cfg.parent
        if looks_gptq(d) and "ncalib128" in str(d).replace("\\", "/").lower() and d not in gptq:
            gptq.append(d)
            print("GPTQ", d)
ranked = sorted(gptq, key=lambda p: (0 if "ungabunga" in str(p).lower() else 1, len(str(p))))
assert ranked, "Attach ungabunga output 346038437 (llama31_8b_base_gptq_int4_ncalib128/)"
INT4_MODEL_ID_OVERRIDE = str(ranked[0])
print("INT4_MODEL_ID_OVERRIDE", INT4_MODEL_ID_OVERRIDE)
"""
)

md("## 1 — Run Cell B (INT4 native) then Cell A (reverse onto FP16)")
code(
    r"""print("=== reverse transfer + native INT4 context K10 ===")
exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"))
print("=== done ===")
"""
)

md("## 2 — Outputs to download")
code(
    r"""from pathlib import Path
out = Path("/kaggle/working/juice_stage2_int4_ncalib128")
print(out)
for p in sorted(out.glob("int4_reverse_and_native_k10*.json")) + sorted(out.glob("steer_*int4_context_k10*.json")):
    print(f"  {p.name:55s} {p.stat().st_size/1024:.1f} KiB")
print()
print("Save Version → download:")
print("  juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_summary.json")
print("  juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_bootstrap.json")
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
out = Path(__file__).resolve().parent / "kaggle_stage2_int4_reverse_native_k10.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
