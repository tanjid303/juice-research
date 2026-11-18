"""Reverse transfer + native INT4 context K10 on the N_CALIB=128 checkpoint.

Cell A — INT4-identified context K10 heads applied to FP16 base.
Cell B — same heads applied to INT4 N_CALIB=128 (in-precision).

Baselines are REUSED (not recomputed):
  FP16: juice_stage2_base_directional_cell3/steer_baseline.json  (fc 16%)
  INT4: this_is_the_128/steer_baseline  (fc 20%; same rows as int4_ncalib128_transfer_bootstrap.json)

Locked: K=10, α+=1, α−=0, 50 ids in split.json seed 42, juice_coherent,
greedy 16 tokens, bootstrap 1000 / seed 42.

Outputs:
  int4_reverse_and_native_k10_summary.json
  int4_reverse_and_native_k10_bootstrap.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE_T0 = time.time()

TAG_A = "steer_reverse_int4_context_k10_on_fp16"
TAG_B = "steer_native_int4_context_k10"
EXPECTED_INT4_K10 = [
    "L7_H26",
    "L14_H10",
    "L7_H21",
    "L4_H5",
    "L3_H3",
    "L31_H20",
    "L4_H11",
    "L14_H23",
    "L8_H29",
    "L9_H0",
]
FP16_K10 = [
    "L4_H9",
    "L9_H0",
    "L1_H11",
    "L4_H5",
    "L19_H27",
    "L17_H12",
    "L7_H26",
    "L14_H23",
    "L3_H22",
    "L31_H20",
]
# Published rates the reused baselines must match.
FP16_BASE_FC = 0.16
INT4_BASE_FC = 0.20
FP16_K10_DELTA_PP = 10.0
INT4_TRANSFER_K10_DELTA_PP = 18.0


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def _bootstrap_repo() -> Path:
    cands = list(Path("/kaggle/working").rglob("common/models.py"))
    cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
    assert cands, "Attach kaggle_juice_stage2_int4_reverse_native_k10.zip (must contain common/)"

    def _juice_attn_ok(models_py: Path) -> bool:
        ja = models_py.parent / "juice_attn.py"
        if not ja.is_file():
            return False
        return "dense_linear_weight_T" in ja.read_text(encoding="utf-8", errors="ignore")

    cands.sort(key=lambda p: (_juice_attn_ok(p), "reverse" in str(p).lower() or "ncalib" in str(p).lower()), reverse=True)
    assert _juice_attn_ok(cands[0]), "common/juice_attn.py missing dense_linear_weight_T"
    repo = cands[0].parent.parent
    dst_common = Path("/kaggle/working/common")
    if dst_common.exists():
        shutil.rmtree(dst_common)
    shutil.copytree(repo / "common", dst_common)
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    print("REPO", repo)
    from common.juice_attn import dense_linear_weight_T  # noqa: F401

    print("juice_attn GPTQ o_proj dequant: OK")
    return repo


def find_one(name: str, prefer: tuple[str, ...] = ()) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name}"

    def score(p: Path) -> tuple:
        s = str(p).replace("\\", "/").lower()
        pref = sum(1 for x in prefer if x.lower() in s)
        return (-pref, 0 if "directional" in s or "native" in s else 1, len(s))

    hits.sort(key=score)
    print(name, "->", hits[0])
    return hits[0]


def _n_calib_of(path: Path) -> int | None:
    meta = path / "quant_meta.json"
    if not meta.is_file():
        return None
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("n_calib"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _dump_kaggle_inputs() -> None:
    print("=== /kaggle/input (top two levels) ===")
    inp = Path("/kaggle/input")
    if not inp.exists():
        print("  MISSING")
        return
    for top in sorted(inp.iterdir()):
        print(f"  {'DIR ' if top.is_dir() else 'FILE'} {top}")
        if top.is_dir():
            kids = sorted(top.iterdir())
            for c in kids[:40]:
                print(f"    {'DIR ' if c.is_dir() else 'FILE'} {c.name}")


def _iter_gptq_dirs() -> list[Path]:
    from common.models import _looks_like_gptq_dir

    found: list[Path] = []
    seen: set[str] = set()
    extra = [
        Path("/kaggle/input/ungabunga/llama31_8b_base_gptq_int4_ncalib128"),
        Path("/kaggle/input/ungabunga/working/llama31_8b_base_gptq_int4_ncalib128"),
    ]
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        extra.append(root / "llama31_8b_base_gptq_int4_ncalib128")
        try:
            for cfg in root.rglob("config.json"):
                extra.append(cfg.parent)
        except OSError as e:
            print("rglob failed on", root, e)
    for d in extra:
        if not d.is_dir():
            continue
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_gptq_dir(d):
            found.append(d)
    return found


def resolve_ncalib128_checkpoint(explicit: str | None = None) -> str:
    from common.models import _looks_like_gptq_dir

    def _ok128(path: Path) -> bool:
        if not _looks_like_gptq_dir(path):
            return False
        s = str(path).replace("\\", "/").lower()
        n = _n_calib_of(path)
        if n == 128 or "ncalib128" in s:
            return True
        if path.name == "llama31_8b_base_gptq_int4" and "ncalib128" not in s:
            return False
        return False

    if explicit:
        p = Path(explicit)
        if _ok128(p):
            print("Resolved N_CALIB=128 GPTQ (override) ->", p, "n_calib", _n_calib_of(p))
            return str(p)

    gptq = _iter_gptq_dirs()
    ranked: list[tuple[int, Path]] = []
    for d in gptq:
        s = str(d).replace("\\", "/").lower()
        n = _n_calib_of(d)
        score = (200 if "ungabunga" in s else 0) + (100 if "ncalib128" in s else 0) + (50 if n == 128 else 0)
        print(f"  n_calib={n}  score={score}  {d}")
        if _ok128(d):
            ranked.append((score, d))
    ranked.sort(key=lambda x: -x[0])
    if ranked:
        picked = ranked[0][1]
        print("Resolved N_CALIB=128 GPTQ ->", picked)
        return str(picked)
    _dump_kaggle_inputs()
    raise FileNotFoundError(
        "N_CALIB=128 checkpoint not found. Attach ungabunga output 346038437."
    )


def label_counts(rows):
    keys = ("parametric", "context", "rejects_context", "other")
    return {k: sum(1 for r in rows if r["label"] == k) for k in keys}


def looks_coherent(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if t.count("://") >= 3:
        return False
    if re.fullmatch(r"[\W\d_]+", t):
        return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))


def coherence_rate(rows):
    return sum(1 for r in rows if looks_coherent(r["text"])) / max(1, len(rows))


def show_examples(rows, title, n_show=5):
    print(f"\n===== {title}: {n_show} raw examples =====")
    for r in rows[:n_show]:
        print(f"- {r['id']} [{r['label']}] {r['text']!r}")


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def _load_baseline(prefer: tuple[str, ...], expect_fc: float, tag: str) -> tuple[list[dict], str]:
    hits: list[Path] = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        try:
            hits.extend(p for p in root.rglob("steer_baseline*.json") if p.is_file())
        except OSError:
            pass

    def ok(p: Path) -> bool:
        if "native_clean_k2" in p.name:
            return False
        if not re.fullmatch(r"steer_baseline(?: \(\d+\))?\.json", p.name):
            return False
        s = str(p).replace("\\", "/").lower()
        return any(x.lower() in s for x in prefer)

    hits = [h for h in hits if ok(h)]
    assert hits, f"Missing reused {tag} baseline ({prefer})"
    hits.sort(key=lambda p: (0 if any(x.lower() in str(p).replace("\\", "/").lower() for x in prefer[:1]) else 1, len(str(p))))
    path = hits[0]
    rows = json.loads(path.read_text(encoding="utf-8"))
    from common.juice import context_follow_rate

    fc = context_follow_rate(rows)
    print(f"reused {tag} baseline -> {path}  n={len(rows)} fc={fc:.3f}")
    assert len(rows) == 50, (tag, len(rows))
    assert abs(fc - expect_fc) < 1e-9, (tag, fc, expect_fc, path)
    return rows, str(path)


def _entry(ident, rows, baseline, boot, extra: dict) -> dict:
    from common.juice import context_follow_rate, head_keys

    r = context_follow_rate(rows)
    r_base = context_follow_rate(baseline)
    entry = {
        "heads": head_keys(ident.enlarge),
        "steer_target": ident.steer_target,
        "K": ident.K,
        "follows_context": r,
        "delta_pp_vs_baseline": (r - r_base) * 100,
        "coherence": coherence_rate(rows),
        "label_counts": label_counts(rows),
        "bootstrap": boot["delta"],
    }
    entry.update(extra)
    return entry


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/tmp/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.51.3",
            "accelerate>=0.26.0,<2.0",
            "safetensors>=0.4.0",
            "gptqmodel==2.2.0",
            "huggingface-hub>=0.26.0",
            "tqdm",
        ]
    )
    elapsed("pip install")

    import torch
    import transformers

    print("torch", torch.__version__, "transformers", transformers.__version__, "ngpu", torch.cuda.device_count())
    assert torch.cuda.is_available()

    repo = _bootstrap_repo()
    try:
        from huggingface_hub import login
        from kaggle_secrets import UserSecretsClient

        login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
        print("HF login ok")
    except Exception as e:
        print("HF login skipped:", type(e).__name__, e)
    elapsed("setup")

    OUT_DIR = Path("/kaggle/working/juice_stage2_int4_ncalib128")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    override = os.environ.get("INT4_MODEL_ID_OVERRIDE") or globals().get("INT4_MODEL_ID_OVERRIDE")
    INT4_MODEL_ID_OVERRIDE = str(override) if override else "/kaggle/working/llama31_8b_base_gptq_int4_ncalib128"

    HEADS_PATH = find_one("heads_int4_native_context_k10.json", prefer=("native_id", "native_clean", "ncalib"))
    SPLIT_PATH = find_one("split.json", prefer=("directional_heads", "base_id"))
    DATA_NAME = "conflictqa_pilot250_popular.jsonl"
    data_hits = list(Path("/kaggle/working").rglob(DATA_NAME))
    data_hits += list(Path("/kaggle/input").rglob(DATA_NAME)) if Path("/kaggle/input").exists() else []
    assert data_hits, "Missing conflictqa_pilot250_popular.jsonl"
    DATA_PATH = data_hits[0]

    from common.conflictqa import load_conflict_items
    from common.juice import (
        apply_juice_steering,
        context_follow_rate,
        head_keys,
        load_identified_heads,
        wrap_llama_attention_for_juice,
    )
    from common.models import load_model_and_tokenizer, unload_model
    from common.stats_analysis import bootstrap_paired_follow_delta

    INT4_MODEL_ID = resolve_ncalib128_checkpoint(INT4_MODEL_ID_OVERRIDE)
    print("INT4_MODEL_ID", INT4_MODEL_ID)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    eval_ids = set(split["eval_ids"])
    eval_items = [it for it in load_conflict_items(DATA_PATH) if it.id in eval_ids]
    assert len(eval_items) == split["n_eval"] == 50
    assert split.get("seed") == 42

    ident = load_identified_heads(HEADS_PATH)
    int4_heads = head_keys(ident.enlarge)
    print("INT4 native ctx K10 ->", int4_heads)
    assert int4_heads == EXPECTED_INT4_K10, int4_heads
    assert ident.K == 10 and ident.steer_target == "context"

    fp16_heads_path = find_one("heads_dir_context_k10.json", prefer=("base_id", "directional_heads"))
    fp16_ident = load_identified_heads(fp16_heads_path)
    fp16_heads = head_keys(fp16_ident.enlarge)
    print("FP16 ctx K10 ->", fp16_heads)
    assert fp16_heads == FP16_K10, fp16_heads

    overlap = sorted(set(int4_heads) & set(fp16_heads), key=lambda k: int4_heads.index(k))
    jac = jaccard(int4_heads, fp16_heads)
    head_sets = {
        "int4_native_context_k10": int4_heads,
        "fp16_context_k10": fp16_heads,
        "int4_heads_source": str(HEADS_PATH),
        "fp16_heads_source": str(fp16_heads_path),
        "overlap": overlap,
        "n_overlap": len(overlap),
        "union_size": len(set(int4_heads) | set(fp16_heads)),
        "jaccard": jac,
        "same_set": set(int4_heads) == set(fp16_heads),
        "note": "Expected Jaccard ~0.33 (5/15). These are NOT the same head set.",
    }
    print("HEAD SETS", json.dumps(head_sets, indent=2))
    assert abs(jac - 1 / 3) < 1e-9, jac
    assert not head_sets["same_set"]

    fp16_base, fp16_base_path = _load_baseline(
        prefer=("fp16_reference", "base_directional_cell3", "juice_stage2_base_directional"),
        expect_fc=FP16_BASE_FC,
        tag="fp16",
    )
    int4_base, int4_base_path = _load_baseline(
        prefer=("ncalib128_reference", "this_is_the_128", "juice_stage2_int4_ncalib128"),
        expect_fc=INT4_BASE_FC,
        tag="int4",
    )
    assert {r["id"] for r in fp16_base} == eval_ids
    assert {r["id"] for r in int4_base} == eval_ids

    ALPHA_PLUS = 1.0
    ALPHA_MINUS = 0.0
    MAX_NEW_TOKENS = 16
    PROMPT_FORMAT = "juice_coherent"
    N_BOOT = 1000
    BOOT_SEED = 42

    # --- Cell B first (INT4, smaller) then Cell A (FP16) ---
    print("\n" + "=" * 70)
    print("CELL B — native INT4 context K10 on N_CALIB=128")
    print("=" * 70)
    print("Loading INT4", INT4_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer("int4", model_id_int4=INT4_MODEL_ID, device_map={"": 0})
    wrap_llama_attention_for_juice(model)
    elapsed("INT4 loaded")
    rows_b = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    boot_b = bootstrap_paired_follow_delta(int4_base, rows_b, n_boot=N_BOOT, seed=BOOT_SEED)
    entry_b = _entry(
        ident,
        rows_b,
        int4_base,
        boot_b,
        {
            "cell": "B_native_int4_context_k10",
            "apply_to": "int4_base_ncalib128",
            "identify_on": "int4_base",
            "baseline_source": int4_base_path,
            "baseline_follows_context": context_follow_rate(int4_base),
            "baseline_coherence": coherence_rate(int4_base),
            "baseline_label_counts": label_counts(int4_base),
            "fp16_identified_k10_delta_pp": FP16_K10_DELTA_PP,
            "int4_transferred_fp16_k10_delta_pp": INT4_TRANSFER_K10_DELTA_PP,
        },
    )
    print(json.dumps({k: v for k, v in entry_b.items() if k != "bootstrap"}, indent=2))
    print("bootstrap", boot_b["delta"]["point_pp"], boot_b["delta"]["ci95_pp"], "excl0", boot_b["delta"]["excludes_zero"])
    show_examples(rows_b, TAG_B)
    (OUT_DIR / f"{TAG_B}.json").write_text(json.dumps(rows_b, indent=2), encoding="utf-8")
    unload_model(model)
    elapsed("CELL B done")

    print("\n" + "=" * 70)
    print("CELL A — reverse transfer: INT4 context K10 heads → FP16")
    print("=" * 70)
    print("Loading FP16 meta-llama/Llama-3.1-8B")
    model, tokenizer = load_model_and_tokenizer(
        "fp16",
        device_map="auto",
        forbid_cpu_offload=True,
    )
    print("device_map", getattr(model, "hf_device_map", None))
    wrap_llama_attention_for_juice(model)
    elapsed("FP16 loaded")
    rows_a = apply_juice_steering(
        model,
        tokenizer,
        eval_items,
        ident,
        alpha_plus=ALPHA_PLUS,
        alpha_minus=ALPHA_MINUS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_format=PROMPT_FORMAT,
    )
    boot_a = bootstrap_paired_follow_delta(fp16_base, rows_a, n_boot=N_BOOT, seed=BOOT_SEED)
    entry_a = _entry(
        ident,
        rows_a,
        fp16_base,
        boot_a,
        {
            "cell": "A_reverse_int4_heads_on_fp16",
            "apply_to": "fp16_base",
            "identify_on": "int4_base",
            "baseline_source": fp16_base_path,
            "baseline_follows_context": context_follow_rate(fp16_base),
            "baseline_coherence": coherence_rate(fp16_base),
            "baseline_label_counts": label_counts(fp16_base),
            "fp16_identified_k10_delta_pp": FP16_K10_DELTA_PP,
            "int4_transferred_fp16_k10_delta_pp": INT4_TRANSFER_K10_DELTA_PP,
        },
    )
    print(json.dumps({k: v for k, v in entry_a.items() if k != "bootstrap"}, indent=2))
    print("bootstrap", boot_a["delta"]["point_pp"], boot_a["delta"]["ci95_pp"], "excl0", boot_a["delta"]["excludes_zero"])
    show_examples(rows_a, TAG_A)
    (OUT_DIR / f"{TAG_A}.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")
    unload_model(model)
    elapsed("CELL A done")

    summary = {
        "cell": "int4_reverse_and_native_k10",
        "n_calib": 128,
        "int4_checkpoint": INT4_MODEL_ID,
        "fp16_checkpoint": "meta-llama/Llama-3.1-8B",
        "prompt_format": PROMPT_FORMAT,
        "alpha_plus": ALPHA_PLUS,
        "alpha_minus": ALPHA_MINUS,
        "n_eval": 50,
        "eval_split_seed": 42,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "head_sets": head_sets,
        "baselines_reused": True,
        "baselines": {
            "fp16": {
                "source": fp16_base_path,
                "follows_context": context_follow_rate(fp16_base),
                "coherence": coherence_rate(fp16_base),
                "label_counts": label_counts(fp16_base),
            },
            "int4_ncalib128": {
                "source": int4_base_path,
                "follows_context": context_follow_rate(int4_base),
                "coherence": coherence_rate(int4_base),
                "label_counts": label_counts(int4_base),
            },
        },
        "comparators": {
            "fp16_identified_k10_on_fp16_delta_pp": FP16_K10_DELTA_PP,
            "fp16_identified_k10_on_int4_n128_delta_pp": INT4_TRANSFER_K10_DELTA_PP,
            "note": "Cell A should be compared to +10 pp (FP16 K10 on FP16). Cell B should be compared to +18 pp (FP16 K10 transferred onto INT4 N128).",
        },
        "conditions": {
            TAG_A: entry_a,
            TAG_B: entry_b,
        },
        "note": (
            "Native INT4 context K10 (Jaccard 0.33 vs FP16 K10, 5 overlapping heads). "
            "Cell A = reverse transfer onto FP16. Cell B = native apply on N_CALIB=128. "
            "Baselines reused from prior sessions; steered rows generated this session."
        ),
    }
    (OUT_DIR / "int4_reverse_and_native_k10_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    payload = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "note": (
            "Resample the 50 eval items with replacement. Delta = steered "
            "follows_context − baseline follows_context, paired on item id. "
            "Baselines reused from prior sessions."
        ),
        "n_calib": 128,
        "checkpoint_int4": INT4_MODEL_ID,
        "checkpoint_fp16": "meta-llama/Llama-3.1-8B",
        "head_sets": head_sets,
        "primary_conditions": [TAG_A, TAG_B],
        "fp16": {
            "precision": "fp16",
            "n_baseline": len(fp16_base),
            "baseline_reused": True,
            "baseline_source": fp16_base_path,
            "conditions": {TAG_A: boot_a},
        },
        "int4": {
            "precision": "int4",
            "n_baseline": len(int4_base),
            "n_calib": 128,
            "baseline_reused": True,
            "baseline_source": int4_base_path,
            "conditions": {TAG_B: boot_b},
        },
    }
    (OUT_DIR / "int4_reverse_and_native_k10_bootstrap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("wrote", OUT_DIR / "int4_reverse_and_native_k10_summary.json")

    print("\n=== TAKEAWAY ===")
    print(f"{'condition':42s} {'fc':>8s} {'delta':>8s} {'CI lo':>8s} {'CI hi':>8s} {'excl0':>7s} {'coh':>6s}")
    print(
        f"{'fp16 baseline (reused)':42s} {context_follow_rate(fp16_base):8.3f} {0.0:8.1f} "
        f"{'':>8s} {'':>8s} {'':>7s} {coherence_rate(fp16_base):6.3f}"
    )
    da = boot_a["delta"]
    print(
        f"{TAG_A:42s} {entry_a['follows_context']:8.3f} {entry_a['delta_pp_vs_baseline']:8.1f} "
        f"{da['ci95_pp'][0]:8.1f} {da['ci95_pp'][1]:8.1f} {str(da['excludes_zero']):>7s} {entry_a['coherence']:6.3f}"
    )
    print(f"{'  comparator: FP16-ID K10 on FP16':42s} {'':8s} {FP16_K10_DELTA_PP:8.1f}")
    print(
        f"{'int4 baseline (reused)':42s} {context_follow_rate(int4_base):8.3f} {0.0:8.1f} "
        f"{'':>8s} {'':>8s} {'':>7s} {coherence_rate(int4_base):6.3f}"
    )
    db = boot_b["delta"]
    print(
        f"{TAG_B:42s} {entry_b['follows_context']:8.3f} {entry_b['delta_pp_vs_baseline']:8.1f} "
        f"{db['ci95_pp'][0]:8.1f} {db['ci95_pp'][1]:8.1f} {str(db['excludes_zero']):>7s} {entry_b['coherence']:6.3f}"
    )
    print(f"{'  comparator: FP16-ID K10 on INT4 N128':42s} {'':8s} {INT4_TRANSFER_K10_DELTA_PP:8.1f}")
    elapsed("done")


if __name__ == "__main__":
    main()
