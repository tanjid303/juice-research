"""Generate juice_repro/kaggle_gemma2_int4_transfer_l27h6.ipynb"""
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
    """# Gemma-2-9B — INT4 transfer of L27_H6 (N_CALIB=128)

Do **not** requantize. Attach the existing `gemma2_9b_base_gptq_int4_ncalib128` checkpoint.

| Locked | Value |
|--------|--------|
| Checkpoint | `gemma2_9b_base_gptq_int4_ncalib128` (N_CALIB=128) |
| Head | **L27_H6** K=1 (FP16 causal effect: −12 pp, CI [−22, −4]) |
| Control | random-K1, seed 42, exclude L27_H6 |
| Eval | same 50 ids (`split.json`, seed 42) |
| Prompt | `juice_coherent`, greedy, 16 tokens, α+=1, α−=0 |
| Stats | paired bootstrap 1000 / seed 42 |

**~1–2.5 h** on GPU T4 ×2 after smoke.

## Before Run All

1. **GPU T4 ×2**, Internet **ON**, secret `HF_TOKEN`
2. Attach **this zip** as a dataset
3. **Add Input → Notebook Output Files** → the **gemma-quantize** version that wrote `gemma2_9b_base_gptq_int4_ncalib128/` (`config.json` + `*.safetensors`). The code zip has no weights.
4. File → Upload `kaggle_gemma2_int4_transfer_l27h6.ipynb`
5. **Run All in the draft.** Do not Save Version first. Quick Save when it prints `SAVE VERSION NOW`.

Do **not** attach `gemma2_gptq_ncalib8`.
"""
)

md("## 0 — Locate scripts + checkpoint")
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
    assert hits, f"Missing {name} — attach kaggle_juice_gemma2_int4_transfer_l27h6.zip"
    hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SCRIPT_02 = find_one("gemma2_02_quant_smoke_test.py")
SCRIPT_03 = find_one("gemma2_03_int4_transfer_l27h6.py")
HEADS = find_one("heads_dir_parametric_k1_raw.json")
SPLIT = find_one("split.json")
print("HEADS", HEADS)
print("SPLIT", SPLIT)

print("\n=== /kaggle/input (top two levels) ===")
inp = Path("/kaggle/input")
if not inp.exists():
    print("  MISSING — attach this zip AND the quantize notebook output")
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

named = "gemma2_9b_base_gptq_int4_ncalib128"
found = []
for root in (Path("/kaggle/working"), Path("/kaggle/input")):
    if not root.exists():
        continue
    try:
        for d in root.rglob(named):
            if looks_gptq(d):
                found.append(d)
                print("FOUND", d)
    except OSError:
        pass
assert found, (
    "Checkpoint not found. Add Input → Notebook Output Files → gemma-quantize "
    "version with gemma2_9b_base_gptq_int4_ncalib128/"
)
os.environ["GEMMA2_INT4_CKPT"] = str(found[0])
print("GEMMA2_INT4_CKPT", os.environ["GEMMA2_INT4_CKPT"])
"""
)

md("## 1 — Quant smoke (JuICE prompts, no requantize)")
code(
    r"""print("=== TASK 1a: gemma2_02_quant_smoke_test ===")
exec(compile(SCRIPT_02.read_text(encoding="utf-8"), str(SCRIPT_02), "exec"))
print("=== smoke done ===")
"""
)

md("## 2 — Transfer L27_H6 + random-K1")
code(
    r"""print("=== TASK 1b: gemma2_03_int4_transfer_l27h6 ===")
exec(compile(SCRIPT_03.read_text(encoding="utf-8"), str(SCRIPT_03), "exec"))
print("=== transfer done ===")
"""
)

md("## 3 — Confirm outputs")
code(
    r"""from pathlib import Path
out = Path("/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6")
for name in (
    "decode_smoke_juice.json",
    "gemma2_int4_transfer_l27h6_summary.json",
    "gemma2_int4_transfer_l27h6_bootstrap.json",
    "steer_parametric_k1_raw.json",
    "random_parametric_k1.json",
):
    p = out / name
    print(name, "ok" if p.is_file() else "MISSING", p)
print()
print("=" * 60)
print("SAVE VERSION NOW — Quick Save, not Save & Run All")
print("Copy the printed gate table back. Do not start other GPTQ cells until then.")
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
try:
    _here = Path(__file__).resolve().parent
    _in_jupyter = False
except NameError:
    _here = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    _in_jupyter = True

out = _here / "kaggle_gemma2_int4_transfer_l27h6.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("Wrote", out)

if _in_jupyter:
    print("Detected Jupyter (no __file__). You uploaded the generator.")
    print("Running Task 1 scripts instead of stopping.")
    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    def _find(name: str):
        hits = []
        for root in roots:
            if root.exists():
                hits.extend(root.rglob(name))
        hits = [h for h in hits if h.is_file()]
        assert hits, name
        return hits[0]
    s02 = _find("gemma2_02_quant_smoke_test.py")
    s03 = _find("gemma2_03_int4_transfer_l27h6.py")
    print("SCRIPT_02", s02)
    print("SCRIPT_03", s03)
    exec(compile(s02.read_text(encoding="utf-8"), str(s02), "exec"))
    exec(compile(s03.read_text(encoding="utf-8"), str(s03), "exec"))

