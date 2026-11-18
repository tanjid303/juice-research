"""Build a Gemma-only Stage-2 n=100 Kaggle package.

This assembles runners and postprocessing code; it does not run inference.
"""
from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"d:\tt")
OUT = ROOT / "kaggle_upload_gemma_n100_verified"
ZIP = ROOT / "kaggle_gemma_n100_verified.zip"

COMMON = [
    "common/__init__.py", "common/models.py", "common/juice.py",
    "common/juice_attn.py", "common/conflictqa.py", "common/scoring.py",
    "common/stats_analysis.py", "common/inference.py", "common/prompts.py",
]


def make_notebook(path: Path, script: str, title: str) -> None:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "gpu"},
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": [f"# {title}\n\n", "Gemma-2-9B base, n=100 expansion.\n",
                         "Protocol: seed 42, 1000 paired bootstrap draws, "
                         "N_CALIB=128 for INT4, greedy `juice_coherent`, "
                         "`max_new_tokens=16`.\n\n",
                         "Run in a fresh Kaggle GPU T4 x2 draft with Internet "
                         "and `HF_TOKEN`. Do not edit the manuscript until all "
                         "five runner outputs and postprocessing are reviewed.\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                 "from pathlib import Path\n",
                 f"hits = list(Path('/kaggle/input').rglob('{script}'))\n",
                 f"hits += list(Path('/kaggle/working').rglob('{script}'))\n",
                 "hits = [p for p in hits if p.is_file()]\n",
                 "assert hits, 'Attach kaggle_gemma_n100_expansion.zip as a dataset'\n",
                 "p = sorted(hits, key=lambda x: len(str(x)))[0]\n",
                 "print('Running', p)\n",
                 "exec(compile(p.read_text(encoding='utf-8'), str(p), 'exec'), "
                 "{'__name__': '__main__'})\n",
             ]},
        ],
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


CLEANUP_SNIPPET = """
import gc
for _name in ('model', 'tokenizer', 'processor', 'qmodel', 'wrapped'):
    globals().pop(_name, None)
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            print(
                'GPU', i,
                'alloc', round(torch.cuda.memory_allocated(i) / 1e9, 2), 'GiB',
                'reserved', round(torch.cuda.memory_reserved(i) / 1e9, 2), 'GiB',
            )
except Exception as _cleanup_error:
    print('CUDA cleanup warning:', _cleanup_error)
""".strip()


def subprocess_runner_cell(script: str, title: str) -> str:
    return f'''# {title} — separate Python process so GPU memory is released on exit
import subprocess, sys
from pathlib import Path
hits = list(Path("/kaggle/input").rglob("{script}"))
hits += list(Path("/kaggle/working").rglob("{script}"))
hits = [p for p in hits if p.is_file()]
assert hits, "Missing {script}; attach the dataset ZIP"
p = sorted(hits, key=lambda x: len(str(x)))[0]
print("Running", p)
subprocess.check_call([sys.executable, "-u", str(p)])
{CLEANUP_SNIPPET}
'''


def make_hminus_battery_notebook(path: Path) -> None:
    """H− only. Headline FP16/INT4 results are already done; do not reload them."""
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Gemma-2-9B n=100 H− battery\n\n",
            "Fresh Kaggle GPU T4 x2 session. Do **not** continue an INT4 headline "
            "session. Each runner is a **subprocess**, so the CUDA context dies "
            "before the next precision load.\n\n",
            "Attach: `kaggle_gemma_n100_verified.zip`, Gemma N_CALIB=128 checkpoint, "
            "Internet, `HF_TOKEN`.\n",
        ]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [line + "\n" for line in '''# Seed completed headline JSON into /kaggle/working (no model load).
import shutil
from pathlib import Path

def seed_dir(name, marker):
    dest = Path("/kaggle/working") / name
    if (dest / marker).is_file():
        print("already present", dest)
        return dest
    hits = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            hits.extend(root.rglob(marker))
    hits = [p.parent for p in hits if p.is_file()]
    hits.sort(key=lambda p: (0 if p.name == name else 1, len(str(p))))
    assert hits, f"Missing {name}/{marker}"
    shutil.copytree(hits[0], dest, dirs_exist_ok=True)
    print("seeded", dest, "from", hits[0])
    return dest

seed_dir("gemma_n100_fp16", "steer_baseline.json")
seed_dir("gemma_n100_int4_headline", "random_context_k10.json")
print("headline JSON seeded; GPU still empty")
'''.splitlines()]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [line + "\n" for line in subprocess_runner_cell(
             "gemma_n100_fp16_hminus.py", "FP16 identified H-").splitlines()]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [line + "\n" for line in subprocess_runner_cell(
             "gemma_n100_fp16_random_hminus.py", "FP16 random H-").splitlines()]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [line + "\n" for line in subprocess_runner_cell(
             "gemma_n100_int4_hminus.py", "INT4 identified and random H-").splitlines()]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [line + "\n" for line in subprocess_runner_cell(
             "gemma_n100_postprocess.py", "Paired table + n=50-style specificity/DiD").splitlines()]},
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "gpu"},
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def make_master_notebook(path: Path) -> None:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Gemma-2-9B n=100 expansion — master runner\n\n",
            "Headline FP16/INT4 n=100 is already complete. Prefer "
            "`gemma_n100_hminus_battery.ipynb` in a **fresh** session for H−.\n\n",
            "If you still run this notebook: every model runner is a subprocess. "
            "Do not `exec` INT4 then load FP16 in the same process.\n",
        ]}
    ]
    fp16_code = """# FP16 baseline, context K10, clean K2, raw L27_H6, random K10
import json
from pathlib import Path
hits = list(Path('/kaggle/input').rglob('gemma_n100_fp16.ipynb'))
hits += list(Path('/kaggle/working').rglob('gemma_n100_fp16.ipynb'))
hits = [p for p in hits if p.is_file()]
assert hits, 'Missing gemma_n100_fp16.ipynb; attach the dataset ZIP'
nb = json.loads(sorted(hits, key=lambda x: len(str(x)))[0].read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), 'gemma_n100_fp16.ipynb', 'exec'), globals())
""" + CLEANUP_SNIPPET
    cells.append({"cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": [line + "\n" for line in fp16_code.splitlines()]})
    for script, title in [
        ("gemma_n100_int4_headline.py", "INT4 headline runner"),
        ("gemma_n100_fp16_hminus.py", "FP16 identified H- runner"),
        ("gemma_n100_fp16_random_hminus.py", "FP16 random H- runner"),
        ("gemma_n100_int4_hminus.py", "INT4 identified/random H- runner"),
        ("gemma_n100_postprocess.py", "Paired raw-table and specificity postprocess"),
    ]:
        cells.append({"cell_type": "code", "metadata": {},
                      "execution_count": None, "outputs": [],
                      "source": [line + "\n" for line in
                                 subprocess_runner_cell(script, title).splitlines()]})
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "gpu"},
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def transform_nb(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    found = [False] * len(replacements)
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        for i, (old, new) in enumerate(replacements):
            if old in text:
                text = text.replace(old, new)
                found[i] = True
        cell["source"] = [line + "\n" for line in text.splitlines()]
    missing = [old[:80] for (old, _), ok in zip(replacements, found) if not ok]
    if missing:
        raise RuntimeError(f"Missing notebook patches in {src}: {missing}")
    dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def n100_script(src: str, out_dir: str) -> str:
    text = (ROOT / "scripts" / src).read_text(encoding="utf-8")
    text = text.replace("== 50", "== 100").replace("same 50", "same 100")
    text = text.replace("n=50", "n=100").replace("50 eval", "100 eval")
    text = text.replace("Resample the 50 eval items", "Resample the 100 eval items")
    text = text.replace('"n_baseline": 50', '"n_baseline": 100')
    text = text.replace(
        'hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))',
        'hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))',
    )
    text = text.replace(
        'hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() or "cell3" in str(p).lower() else 1, len(str(p))))',
        'hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))',
    ).replace(
        'rand_path_hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() or "cell3" in str(p).lower() else 1, len(str(p))))',
        'rand_path_hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))',
    )
    text = text.replace(
        'rand_path_hits = [p for p in rand_path_hits if p.is_file()]',
        '''rand_path_hits = [p for p in rand_path_hits if p.is_file()]
    _required_random_heads = {
        "L41_H11", "L7_H3", "L1_H9", "L17_H13", "L15_H13",
        "L14_H7", "L9_H1", "L6_H9", "L35_H9", "L5_H10",
    }
    rand_path_hits = [
        p for p in rand_path_hits
        if set(json.loads(p.read_text(encoding="utf-8")).get("heads", []))
        == _required_random_heads
    ]''',
    )
    # Every source runner uses one of these output-directory literals.
    text = text.replace(
        'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_fp16_context_suppress")',
        f'OUT_DIR = Path("/kaggle/working/{out_dir}")',
    ).replace(
        'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_fp16_random_k10_suppress")',
        f'OUT_DIR = Path("/kaggle/working/{out_dir}")',
    ).replace(
        'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_int4_context_suppress")',
        f'OUT_DIR = Path("/kaggle/working/{out_dir}")',
    )
    text = text.replace(
        'SPLIT_PATH = find_one("split.json")',
        '''split_candidates = []
    for _root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if _root.exists():
            split_candidates.extend(_root.rglob("split.json"))
    SPLIT_PATH = next(
        (p for p in split_candidates
         if json.loads(p.read_text(encoding="utf-8")).get("n_eval") == 100),
        None,
    )
    assert SPLIT_PATH is not None, "Missing n=100 split.json"''',
    )
    text = text.replace(
        'RAND_PATH = find_one("random_k10_heads.json")',
        '''_random_candidates = []
    for _root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if _root.exists():
            _random_candidates.extend(_root.rglob("random_k10_heads.json"))
    _required_random_heads = {
        "L41_H11", "L7_H3", "L1_H9", "L17_H13", "L15_H13",
        "L14_H7", "L9_H1", "L6_H9", "L35_H9", "L5_H10",
    }
    _random_candidates = [
        p for p in _random_candidates
        if p.is_file()
        and set(json.loads(p.read_text(encoding="utf-8")).get("heads", []))
        == _required_random_heads
    ]
    assert _random_candidates, "Missing exact locked random-K10 head set"
    RAND_PATH = _random_candidates[0]''',
    )
    text = text.replace(
        "from common.models import gpu_max_memory, load_model_and_tokenizer, unload_model",
        "from common.models import force_release_cuda, gpu_max_memory, load_model_and_tokenizer, unload_model",
    ).replace(
        "from common.models import load_model_and_tokenizer, resolve_gemma2_ncalib128_checkpoint, unload_model",
        "from common.models import force_release_cuda, load_model_and_tokenizer, resolve_gemma2_ncalib128_checkpoint, unload_model",
    )
    text = text.replace(
        'print("Loading", MODEL_ID)',
        'force_release_cuda("before FP16 load")\n    print("Loading", MODEL_ID)',
    ).replace(
        'print("Loading INT4 (no requantize)", INT4_MODEL_ID)',
        'force_release_cuda("before INT4 load")\n    print("Loading INT4 (no requantize)", INT4_MODEL_ID)',
    )
    text = text.replace("unload_model(model)", "unload_model(model, tokenizer)")
    text = text.replace(
        '''    else:
        print("WARNING: random_k10_heads.json missing; will resample seed=42 exclude ctx K10 only after load")
        RAND_NOTE = "Fallback resample seed=42, exclude identified context K10 only (Cell-3 draw missing)."
        rand_keys = None
''',
        '''    else:
        raise AssertionError(
            "Missing exact locked random-K10 set "
            "L41_H11 L7_H3 L1_H9 L17_H13 L15_H13 L14_H7 L9_H1 L6_H9 L35_H9 L5_H10"
        )
''',
    )
    text = text.replace(
        '''    if rand_keys is None:
        from common.juice import model_dims, random_head_control

        n_layers, n_heads, _ = model_dims(model)
        rand_heads = random_head_control(n_layers, n_heads, 10, seed=42, exclude=ident_sup.suppress)
        rand_keys = head_keys(rand_heads)
''',
        '''    assert rand_keys == [
        "L41_H11", "L7_H3", "L1_H9", "L17_H13", "L15_H13",
        "L14_H7", "L9_H1", "L6_H9", "L35_H9", "L5_H10",
    ], rand_keys
''',
    )
    return text


def make_int4_headline() -> str:
    text = (ROOT / "scripts" / "gemma2_03_int4_transfer_l27h6.py").read_text(
        encoding="utf-8"
    )
    text = text.replace(
        'OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6")',
        'OUT_DIR = Path("/kaggle/working/gemma_n100_int4_headline")',
    ).replace("== 50", "== 100").replace("Same 50", "Same 100").replace("n=50", "n=100")
    text = text.replace("Resample the 50 eval items", "Resample the 100 eval items")
    text = text.replace('"n_baseline": 50', '"n_baseline": 100')
    text = text.replace(
        'hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))',
        'hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))',
    )
    text = text.replace(
        'hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() or "cell3" in str(p).lower() else 1, len(str(p))))',
        'hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))',
    )
    text = text.replace(
        'HEADS_PATH = find_one("heads_dir_parametric_k1_raw.json")',
        '''RAW_HEADS_PATH = find_one("heads_dir_parametric_k1_raw.json")
    CTX_HEADS_PATH = find_one("heads_dir_context_k10.json")''',
    )
    text = text.replace(
        'ident = load_identified_heads(HEADS_PATH)\n    assert head_keys(ident.enlarge) == ["L27_H6"], head_keys(ident.enlarge)',
        '''ident = load_identified_heads(RAW_HEADS_PATH)
    ident_ctx = load_identified_heads(CTX_HEADS_PATH)
    assert head_keys(ident.enlarge) == ["L27_H6"], head_keys(ident.enlarge)
    assert len(ident_ctx.enlarge) == 10, head_keys(ident_ctx.enlarge)''',
    )
    old = '''    conditions = [
        ("steer_parametric_k1_raw", ident),
        ("random_parametric_k1", ident_rand),
    ]'''
    new = '''    ctx_rand_heads = [
        HeadRef(41, 11), HeadRef(7, 3), HeadRef(1, 9), HeadRef(17, 13),
        HeadRef(15, 13), HeadRef(14, 7), HeadRef(9, 1), HeadRef(6, 9),
        HeadRef(35, 9), HeadRef(5, 10),
    ]
    assert head_keys(ctx_rand_heads) == [
        "L41_H11", "L7_H3", "L1_H9", "L17_H13", "L15_H13",
        "L14_H7", "L9_H1", "L6_H9", "L35_H9", "L5_H10",
    ]
    ident_ctx_rand = identified_from_enlarge(
        ctx_rand_heads, precision="int4_gemma2_random_context", K=10,
        steer_target="context"
    )
    (OUT_DIR / "random_k10_heads.json").write_text(
        json.dumps({"heads": head_keys(ctx_rand_heads), "seed": RAND_SEED,
                    "excluded": head_keys(ident_ctx.enlarge)}, indent=2),
        encoding="utf-8"
    )
    conditions = [
        ("steer_context_k10", ident_ctx),
        ("random_context_k10", ident_ctx_rand),
        ("steer_parametric_k1_raw", ident),
        ("random_parametric_k1", ident_rand),
    ]'''
    if old not in text:
        raise RuntimeError("Could not locate INT4 Gemma headline conditions")
    return text.replace(old, new).replace(
        'SPLIT_PATH = find_one("split.json")',
        '''split_candidates = []
    for _root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if _root.exists():
            split_candidates.extend(_root.rglob("split.json"))
    SPLIT_PATH = next(
        (p for p in split_candidates
         if json.loads(p.read_text(encoding="utf-8")).get("n_eval") == 100),
        None,
    )
    assert SPLIT_PATH is not None, "Missing n=100 split.json"''',
    )


def make_postprocess() -> str:
    return (ROOT / "scripts" / "gemma_n100_postprocess.py").read_text(encoding="utf-8")

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in COMMON:
        dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    data = OUT / "data"; data.mkdir()
    shutil.copy2(ROOT / "data/conflictqa_pilot250_popular.jsonl", data / "conflictqa_pilot250_popular.jsonl")
    ids = [json.loads(x)["id"] for x in (ROOT/"data/conflictqa_pilot250_popular.jsonl").read_text(encoding="utf-8").splitlines() if x]
    random.Random(42).shuffle(ids)
    heads = OUT / "directional_heads"; heads.mkdir()
    split = {"seed":42,"n_calib":25,"n_eval":100,"calib_ids":sorted(ids[:25]),"eval_ids":sorted(ids[25:125])}
    (heads/"split.json").write_text(json.dumps(split,indent=2))
    src_heads = ROOT / "juice_stage2_gemma2_base_id/directional_heads"
    for name in ["heads_dir_context_k10.json","heads_dir_parametric_k2.json",
                 "heads_dir_parametric_k2_clean.json","heads_dir_parametric_k1_raw.json"]:
        shutil.copy2(src_heads/name, heads/name)
    (heads/"random_k10_heads.json").write_text(json.dumps({
        "heads": ["L41_H11", "L7_H3", "L1_H9", "L17_H13", "L15_H13",
                  "L14_H7", "L9_H1", "L6_H9", "L35_H9", "L5_H10"],
        "seed": 42,
        "excluded": ["L40_H14", "L17_H7", "L24_H15", "L22_H0", "L32_H11",
                     "L25_H0", "L3_H9", "L8_H9", "L7_H6", "L29_H5",
                     "L31_H7", "L39_H6", "L27_H6"],
    }, indent=2), encoding="utf-8")
    scripts = OUT/"scripts"; scripts.mkdir()
    (scripts/"gemma_n100_int4_headline.py").write_text(make_int4_headline(),encoding="utf-8")
    for src, dst, outdir in [
        ("gemma2_04_fp16_context_suppress_check.py","gemma_n100_fp16_hminus.py","gemma_n100_fp16_hminus"),
        ("gemma2_06_fp16_random_k10_suppress.py","gemma_n100_fp16_random_hminus.py","gemma_n100_fp16_random_hminus"),
        ("gemma2_05_int4_context_suppress_transfer.py","gemma_n100_int4_hminus.py","gemma_n100_int4_hminus"),
    ]:
        (scripts/dst).write_text(n100_script(src,outdir),encoding="utf-8")
    # Reuse the existing FP16 Cell-3 notebook and patch only its evaluation size/output.
    repro = OUT/"notebooks"; repro.mkdir()
    src_nb = ROOT/"juice_repro/kaggle_gemma2_base_cell3.ipynb"
    transform_nb(src_nb,repro/"gemma_n100_fp16.ipynb",[
        ('OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_base_cell3")','OUT_DIR = Path("/kaggle/working/gemma_n100_fp16")'),
        ('hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))',
         'hits.sort(key=lambda p: (0 if "gemma_n100_expansion" in str(p).lower() or "gemma_n100" in str(p).lower() else 1, len(str(p))))'),
        ('SPLIT_PATH = find_one("split.json")',
         '''split_candidates = []
for _root in [Path("/kaggle/working"), Path("/kaggle/input")]:
    if _root.exists():
        split_candidates.extend(_root.rglob("split.json"))
SPLIT_PATH = next(
    (p for p in split_candidates
     if json.loads(p.read_text(encoding="utf-8")).get("n_eval") == 100),
    None,
)
assert SPLIT_PATH is not None, "Missing n=100 split.json"'''),
        ('assert len(eval_items) == split["n_eval"] == 50','assert len(eval_items) == split["n_eval"] == 100'),
        ("same 50 IDs","same 100 IDs"),("50 eval","100 eval")
    ])
    make_notebook(repro/"gemma_n100_int4_headline.ipynb","gemma_n100_int4_headline.py","Gemma INT4 n=100 headline")
    make_notebook(repro/"gemma_n100_fp16_hminus.ipynb","gemma_n100_fp16_hminus.py","Gemma FP16 identified H-")
    make_notebook(repro/"gemma_n100_fp16_random_hminus.ipynb","gemma_n100_fp16_random_hminus.py","Gemma FP16 random H-")
    make_notebook(repro/"gemma_n100_int4_hminus.ipynb","gemma_n100_int4_hminus.py","Gemma INT4 identified and random H-")
    (scripts/"gemma_n100_postprocess.py").write_text(make_postprocess(),encoding="utf-8")
    make_notebook(repro/"gemma_n100_postprocess.ipynb","gemma_n100_postprocess.py","Gemma n=100 paired raw table and specificity")
    make_master_notebook(OUT/"gemma_n100_expansion_master.ipynb")
    make_hminus_battery_notebook(OUT/"gemma_n100_hminus_battery.ipynb")
    make_hminus_battery_notebook(repro/"gemma_n100_hminus_battery.ipynb")
    shutil.copy2(OUT/"gemma_n100_hminus_battery.ipynb", ROOT/"kaggle_gemma_n100_hminus_battery.ipynb")
    headline = OUT / "headline_results"
    shutil.copytree(ROOT / "results-again-fail" / "gemma_n100_fp16", headline / "gemma_n100_fp16")
    shutil.copytree(ROOT / "results-again-fail" / "gemma_n100_int4_headline", headline / "gemma_n100_int4_headline")
    (OUT/"README.md").write_text("""# Gemma-2-9B n=100 expansion

Headline FP16/INT4 n=100 is already complete and included under `headline_results/`.
Do **not** rerun those cells.

## H- battery (this is the remaining work)

1. New Kaggle GPU T4 x2 draft. Factory reset if a previous INT4 session is still live.
2. Attach this ZIP, the Gemma N_CALIB=128 checkpoint, Internet, `HF_TOKEN`.
3. Upload and run `gemma_n100_hminus_battery.ipynb` **only**.
4. Each H- runner is a **subprocess**. The CUDA context dies before the next load.
5. Download `/kaggle/working/gemma_n100_fp16_hminus/`, `gemma_n100_fp16_random_hminus/`,
   `gemma_n100_int4_hminus/`, `gemma_n100_raw_results.json`, and
   `gemma_n100_hminus_specificity_did.json`.

Locked random-K10 (must appear in every random-control log):
L41_H11, L7_H3, L1_H9, L17_H13, L15_H13, L14_H7, L9_H1, L6_H9, L35_H9, L5_H10

Protocol: seed 42, 100 eval IDs, 1000 paired bootstrap draws, juice_coherent, 16 tokens.
Do not edit the paper until those five outputs are reviewed.
""",encoding="utf-8")
    if ZIP.exists(): ZIP.unlink()
    with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(OUT).as_posix())
    print("Wrote",ZIP,ZIP.stat().st_size)

if __name__ == "__main__":
    main()
