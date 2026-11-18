"""Build the preregistered Stage-2 n=100 headline Kaggle package.

This builder only assembles Kaggle code and artifacts. It does not run model
inference, quantization, or any evaluation.
"""
from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_stage2_n100_headline"
ZIP = ROOT / "kaggle_stage2_n100_headline.zip"


def write_notebook(path: Path, title: str, script_name: str, checkpoint_note: str) -> None:
    setup = f"""# {title}

This notebook runs only the preregistered Stage-2 n=100 headline cells.
It does not run n=150 or the repair suite.

{checkpoint_note}

Before Run All: use GPU T4 x2, enable Internet, attach this package as a
dataset, and add the HF_TOKEN secret when the model is gated.
"""
    code = f"""from pathlib import Path

hits = list(Path("/kaggle/working").rglob("{script_name}"))
hits += list(Path("/kaggle/input").rglob("{script_name}"))
hits = [p for p in hits if p.is_file()]
assert hits, "Attach kaggle_stage2_n100_headline.zip as a dataset"
script = sorted(hits, key=lambda p: ("stage2_n100" not in str(p), len(str(p))))[0]
print("Running", script)
exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), {{"__name__": "__main__"}})
"""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "gpu",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in setup.splitlines()],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ],
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def transform_notebook(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    found = [False] * len(replacements)
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        for i, (old, new) in enumerate(replacements):
            if old in text:
                text = text.replace(old, new)
                found[i] = True
        if "kaggle_n100_gemma_fp16" in dst.name and 'print("\\n=== READ ===")' in text:
            start = text.index('print("\\n=== READ ===")')
            end = text.index('print("Do not start GPTQ', start)
            read_block = '''print("\\n=== READ ===")
raw = summary["conditions"]["steer_parametric_k1_raw"]
raw_boot = boot_payload["fp16"]["conditions"]["steer_parametric_k1_raw"]
print(
    f"  raw par peak L27_H6 K=1: d={raw['delta_pp_vs_baseline']:+.1f} pp "
    f"CI={fmt_ci(raw_boot)} "
    f"excl0={raw['bootstrap']['excludes_zero']} P/C/O={pco(raw['label_counts'])}"
)

'''
            text = text[:start] + read_block + text[end:]
        cell["source"] = [line + "\n" for line in text.splitlines()]
    missing = [old[:100] for (old, _), ok in zip(replacements, found) if not ok]
    if missing:
        raise RuntimeError(f"Could not find notebook patches in {src}: {missing!r}")
    dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def make_n100_split() -> dict:
    source = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
    ids = [
        json.loads(line)["id"]
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ids) == 250, len(ids)
    random.Random(42).shuffle(ids)
    return {
        "seed": 42,
        "n_calib": 25,
        "n_eval": 100,
        "calib_ids": sorted(ids[:25]),
        "eval_ids": sorted(ids[25:125]),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    common = [
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
    for rel in common:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    data_dst = OUT / "data" / "conflictqa_pilot250_popular.jsonl"
    data_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "data" / "conflictqa_pilot250_popular.jsonl", data_dst)

    split = make_n100_split()
    heads = OUT / "directional_heads"
    heads.mkdir(parents=True, exist_ok=True)
    (heads / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    head_names = [
        "heads_dir_context_k10.json",
        "heads_dir_context_k5.json",
        "heads_dir_parametric_k2.json",
        "heads_dir_parametric_k3.json",
        "heads_dir_parametric_k2_clean.json",
        "heads_dir_parametric_k1_raw.json",
    ]
    for name in head_names:
        src = ROOT / "juice_stage2_base_id" / "directional_heads" / name
        if not src.is_file():
            src = ROOT / "juice_stage2_gemma2_base_id" / "directional_heads" / name
        shutil.copy2(src, heads / name)

    repro = OUT / "juice_repro"
    repro.mkdir(parents=True)

    # FP16 Llama: baseline + context K10 + clean parametric K2 + random K10.
    llama_fp16_repls = [
        ('OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_base_cell3")',
         'OUT_DIR = Path("/kaggle/working/juice_stage2_llama_base_n100")'),
        ('MODEL_ID = "google/gemma-2-9b"', 'MODEL_ID = "meta-llama/Llama-3.1-8B"'),
        ('assert len(eval_items) == split["n_eval"] == 50',
         'assert len(eval_items) == split["n_eval"] == 100'),
        ('assert head_keys(ident_par.enlarge) == head_keys(ident_par_dir.enlarge) == ["L31_H7", "L39_H6"]',
         'assert head_keys(ident_par.enlarge) == ["L20_H14", "L26_H15"]'),
        ('assert head_keys(ident_raw.enlarge) == ["L27_H6"]',
         'print("par_k1_raw (loaded for artifact compatibility):", head_keys(ident_raw.enlarge))'),
        ('assert model.config.model_type == "gemma2"', 'assert model.config.model_type == "llama"'),
        ('identified_exclude = list(ident_ctx.enlarge) + list(ident_par.enlarge) + list(ident_raw.enlarge)',
         'identified_exclude = list(ident_ctx.enlarge)'),
        ('precision="fp16_gemma2_random"', 'precision="fp16_llama_random"'),
        ('''conditions = [
    ("steer_context_k10", ident_ctx),
    ("steer_parametric_k2_clean", ident_par),
    ("steer_parametric_k1_raw", ident_raw),
    ("random_context_k10", ident_rand),
]''',
         '''conditions = [
    ("steer_context_k10", ident_ctx),
    ("steer_parametric_k2_clean", ident_par),
    ("random_context_k10", ident_rand),
]'''),
        ('''"steer_context_k10",
        "steer_parametric_k2_clean",
        "steer_parametric_k1_raw",
        "random_context_k10",''',
         '''"steer_context_k10",
        "steer_parametric_k2_clean",
        "random_context_k10",'''),
        ("Resample the 50 eval items", "Resample the 100 eval items"),
        ("gemma2_base_cell3_summary.json", "llama_base_n100_summary.json"),
        ("gemma2_base_cell3_bootstrap.json", "llama_base_n100_bootstrap.json"),
    ]
    transform_notebook(
        ROOT / "juice_repro" / "kaggle_gemma2_base_cell3.ipynb",
        repro / "kaggle_n100_llama_fp16.ipynb",
        llama_fp16_repls,
    )

    # FP16 Gemma: baseline + raw L27_H6 only.
    gemma_fp16_repls = [
        ('OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_base_cell3")',
         'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma_base_n100")'),
        ('assert len(eval_items) == split["n_eval"] == 50',
         'assert len(eval_items) == split["n_eval"] == 100'),
        ('''conditions = [
    ("steer_context_k10", ident_ctx),
    ("steer_parametric_k2_clean", ident_par),
    ("steer_parametric_k1_raw", ident_raw),
    ("random_context_k10", ident_rand),
]''',
         '''conditions = [
    ("steer_parametric_k1_raw", ident_raw),
 ]'''),
        ('''"steer_context_k10",
        "steer_parametric_k2_clean",
        "steer_parametric_k1_raw",
        "random_context_k10",''',
         '''"steer_parametric_k1_raw",'''),
        ("Resample the 50 eval items", "Resample the 100 eval items"),
        ("gemma2_base_cell3_summary.json", "gemma_base_n100_summary.json"),
        ("gemma2_base_cell3_bootstrap.json", "gemma_base_n100_bootstrap.json"),
    ]
    transform_notebook(
        ROOT / "juice_repro" / "kaggle_gemma2_base_cell3.ipynb",
        repro / "kaggle_n100_gemma_fp16.ipynb",
        gemma_fp16_repls,
    )

    # INT4 Llama: use the already validated N_CALIB=128 transfer runner,
    # which includes baseline, context K10, random K10, and clean K2.
    llama_script = (
        ROOT / "scripts" / "ncalib128_02_rerun_three_cells.py"
    ).read_text(encoding="utf-8")
    llama_script = llama_script.replace(
        'OUT_DIR = Path("/kaggle/working/juice_stage2_int4_ncalib128")',
        'OUT_DIR = Path("/kaggle/working/juice_stage2_int4_ncalib128_n100")',
    ).replace("same 50 eval ids", "same 100 eval ids").replace(
        "n=50", "n=100"
    )
    scripts = OUT / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "llama_int4_n100_headline.py").write_text(llama_script, encoding="utf-8")
    write_notebook(
        repro / "kaggle_n100_llama_int4.ipynb",
        "Llama-3.1-8B base GPTQ INT4 — n=100 headline cells",
        "llama_int4_n100_headline.py",
        "Attach the existing N_CALIB=128 Llama GPTQ checkpoint output containing "
        "`llama31_8b_base_gptq_int4_ncalib128/`. Do not requantize.",
    )

    # INT4 Gemma: raw L27_H6 plus random-K1, with dynamic n_eval assertion.
    gemma_script = (
        ROOT / "scripts" / "gemma2_03_int4_transfer_l27h6.py"
    ).read_text(encoding="utf-8")
    gemma_script = gemma_script.replace(
        'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6")',
        'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6_n100")',
    ).replace(
        'assert len(eval_items) == split["n_eval"] == 50,',
        'assert len(eval_items) == split["n_eval"] == 100,',
    ).replace("Same 50 eval ids", "Same 100 eval ids").replace("n=50", "n=100")
    (scripts / "gemma_int4_n100_headline.py").write_text(gemma_script, encoding="utf-8")
    write_notebook(
        repro / "kaggle_n100_gemma_int4.ipynb",
        "Gemma-2-9B base GPTQ INT4 — n=100 headline cells",
        "gemma_int4_n100_headline.py",
        "Attach the existing N_CALIB=128 Gemma GPTQ checkpoint output containing "
        "`gemma2_9b_base_gptq_int4_ncalib128/`. Do not requantize.",
    )

    rule = ROOT / "expansion_selection_rule.md"
    shutil.copy2(rule, OUT / rule.name)

    readme = """# Stage 2 n=100 headline Kaggle package

This package runs only the preregistered n=100 evaluation set:
`random.Random(42)` over the full 250-item pool, with positions `25:125`
as the 100 evaluation IDs. It does not run n=150 or the repair suite.

## Four notebooks

Run each notebook in a fresh Kaggle draft with GPU T4 x2, Internet enabled,
and `HF_TOKEN` configured:

1. `juice_repro/kaggle_n100_llama_fp16.ipynb`
   - Llama FP16 baseline, context K10, clean parametric K2, random K10.
2. `juice_repro/kaggle_n100_llama_int4.ipynb`
   - Llama N_CALIB=128 INT4 baseline, context K10 transfer, random K10,
     clean parametric K2 transfer.
3. `juice_repro/kaggle_n100_gemma_fp16.ipynb`
   - Gemma FP16 baseline and raw L27_H6.
4. `juice_repro/kaggle_n100_gemma_int4.ipynb`
   - Gemma N_CALIB=128 INT4 baseline, raw L27_H6 transfer, random K1.

The INT4 notebooks require the existing N_CALIB=128 checkpoint as an
additional Notebook Output input. They do not requantize.

Every runner uses `juice_coherent`, greedy decoding, 16 new tokens,
alpha+=1, alpha-=0, paired bootstrap with 1000 draws and seed 42.

Download the JSON files from:

- `/kaggle/working/juice_stage2_llama_base_n100/`
- `/kaggle/working/juice_stage2_int4_ncalib128_n100/`
- `/kaggle/working/juice_stage2_gemma_base_n100/`
- `/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6_n100/`

Do not start n=150 until the n=100 outputs have been reviewed.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT).as_posix())

    print(f"Wrote {OUT}")
    print(f"Wrote {ZIP} ({ZIP.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
