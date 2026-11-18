"""Generate juice_repro/kaggle_stage2_cell_a_alpha_sweep.ipynb"""
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
    """# Cell A diagnostic — α-sweep INT4 K10 → FP16

α+ ∈ {0.25, 0.5, 1.0}, α−=0, same 50 ids, reused FP16 baseline (fc 16%).
α=1 is reused from the reverse-transfer JSON bundled in this zip (not regenerated).

No INT4 checkpoint. **GPU T4 ×2**, Internet ON, `HF_TOKEN`. ~2–3 h.

A genuine scaled circuit should shrink toward ~+10 pp at low α, stay coherent, and drop the exam-meta dump. A stuck +40-plus with “Note: multiple-choice…” at α=0.25 is overdrive, not FP16’s K10 object.
"""
)
md("## 0 — Locate")
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
    assert hits, f"Missing {name}"
    hits.sort(key=lambda p: (0 if "alpha" in str(p).lower() or "reverse" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT = find_one("ncalib128_06_cell_a_alpha_sweep.py")
print("SCRIPT", SCRIPT)
"""
)
md("## 1 — Run")
code(
    r"""print("=== Cell A α-sweep ===")
exec(compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"))
print("=== done ===")
"""
)
md("## 2 — Download")
code(
    r"""from pathlib import Path
out = Path("/kaggle/working/juice_stage2_int4_ncalib128")
for p in sorted(out.glob("cell_a_alpha_sweep*.json")) + sorted(out.glob("steer_reverse_int4_k10_fp16_a*.json")):
    print(f"  {p.name:50s} {p.stat().st_size/1024:.1f} KiB")
print("Download cell_a_alpha_sweep_summary.json")
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
out = Path(__file__).resolve().parent / "kaggle_stage2_cell_a_alpha_sweep.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
