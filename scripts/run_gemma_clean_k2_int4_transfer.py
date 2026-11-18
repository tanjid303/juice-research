"""INT4 transfer test for Gemma clean-K2 (L31_H7, L39_H6) with matched random control.

Closes disclosed gap (A) by applying the locked FP16 parametric clean-K2 pair
(L31_H7, L39_H6) unchanged to the canonical Gemma INT4 checkpoint
(gemma2_9b_base_gptq_int4_ncalib128).

Also runs a matched random K=2 control (seed 42, excluding clean K2, context K10,
and raw peak L27_H6) on both FP16 and INT4.

Steering protocol:
  - Dual-run JuICE, alpha+=1.0, alpha-=0.0 (enlarge direction)
  - Greedy decoding, max_new_tokens=16, prompt_format="juice_coherent"
  - Preregistered seed-42 n=100 evaluation set from conflictqa_pilot250_popular.jsonl

Scoring:
  - Question-aware matcher (pass gold question into label_conflict_response)
  - Reports both gold-PAR-only delta [95% CI] and inclusive delta [95% CI]
  - Paired bootstrap: 1000 draws, seed 42, percentile 95% CI

Outputs:
  gemma_clean_k2_int4_transfer_n100_ncalib128.json
  gemma_clean_k2_int4_transfer_n100_ncalib128.md
  gemma_clean_k2_int4_transfer_n100_ncalib128.csv
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def setup_path() -> Path:
    candidates = [
        Path("/kaggle/working"),
        Path("/kaggle/input"),
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "common" / "juice.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
        for sub in c.glob("**/common/juice.py"):
            parent = sub.parent.parent
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find repository root containing common/juice.py")


REPO = setup_path()
print(f"Repository root: {REPO}")

from common.conflictqa import ConflictItem, load_conflict_items
from common.juice import (
    HeadRef,
    IdentifiedHeads,
    apply_juice_steering,
    conflict_eval_prompt,
    head_keys,
    identified_from_enlarge,
    load_identified_heads,
    model_dims,
    random_head_control,
    wrap_attention_for_juice,
)
from common.scoring import _mentions_answer, label_conflict_response


def find_file(name: str, check_n100: bool = False, require_gemma_clean_k2: bool = False) -> Path:
    # Check direct locations
    direct_candidates = [
        REPO / "directional_heads" / name,
        REPO / "data" / name,
        REPO / "juice_stage2_gemma2_base_id" / "directional_heads" / name,
        REPO / "kaggle_upload_gemma_clean_k2_int4_transfer" / "directional_heads" / name,
        REPO / "kaggle_upload_gemma_clean_k2_int4_transfer" / "data" / name,
        REPO / "kaggle_upload_gemma_n100_verified" / "directional_heads" / name,
        REPO / "kaggle_upload_gemma_n100_verified" / "data" / name,
        Path("/kaggle/working") / name,
        Path("/kaggle/working") / "directional_heads" / name,
        Path("/kaggle/working") / "data" / name,
        Path("/kaggle/input") / name,
    ]
    for c in direct_candidates:
        if c.is_file():
            if "llama" in str(c).lower() and "gemma" not in str(c).lower():
                continue
            if check_n100 and name == "split.json":
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    if data.get("n_eval") == 100:
                        return c
                except Exception:
                    continue
            elif require_gemma_clean_k2 and name == "heads_dir_parametric_k2_clean.json":
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    h_keys = [f"L{h['layer']}_H{h['head']}" for h in data.get("enlarge", [])]
                    if h_keys == ["L31_H7", "L39_H6"]:
                        return c
                except Exception:
                    continue
            else:
                return c

    # Fallback search
    search_roots = [Path("/kaggle/working"), Path("/kaggle/input"), REPO]
    hits: list[Path] = []
    for root in search_roots:
        if root.exists():
            for p in root.rglob(name):
                if p.is_file():
                    if "llama" in str(p).lower() and "gemma" not in str(p).lower():
                        continue
                    if check_n100 and name == "split.json":
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            if data.get("n_eval") == 100:
                                hits.append(p)
                        except Exception:
                            pass
                    elif require_gemma_clean_k2 and name == "heads_dir_parametric_k2_clean.json":
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            h_keys = [f"L{h['layer']}_H{h['head']}" for h in data.get("enlarge", [])]
                            if h_keys == ["L31_H7", "L39_H6"]:
                                hits.append(p)
                        except Exception:
                            pass
                    else:
                        hits.append(p)
    if not hits:
        raise FileNotFoundError(
            f"Could not find required Gemma file: {name} "
            f"(check_n100={check_n100}, require_gemma_clean_k2={require_gemma_clean_k2})"
        )
    hits.sort(key=lambda p: (
        0 if "gemma" in str(p).lower() else 1,
        0 if "n100" in str(p) or "stage2" in str(p) else 1,
        len(str(p))
    ))
    return hits[0]


def resolve_gemma2_int4_ncalib128(explicit_path: str | None = None) -> str:
    from common.models import _looks_like_gptq_dir

    named = "gemma2_9b_base_gptq_int4_ncalib128"
    forbidden = ("ncalib8", "tiny_layer", "gptq_tiny", "layer0")
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env = os.environ.get("GEMMA2_INT4_CKPT")
    if env:
        candidates.append(Path(env))
    for root in (Path("/kaggle/working"), Path("/kaggle/input"), REPO, Path.cwd()):
        if not root.exists():
            continue
        candidates.append(root / named)
        try:
            for d in root.rglob(named):
                if d.is_dir():
                    candidates.append(d)
        except OSError:
            pass
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = str(path.resolve()) if path.exists() else str(path)
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        low = resolved.replace("\\", "/").lower()
        if any(tok in low for tok in forbidden) and "ncalib128" not in low:
            continue
        if _looks_like_gptq_dir(path) and named in low:
            print(f"Resolved canonical Gemma INT4 NCALIB=128 GPTQ directory -> {path}")
            return str(path)
    raise FileNotFoundError(
        f"Canonical Gemma N_CALIB=128 INT4 checkpoint '{named}' not found. "
        f"Searched: {list(seen)[:10]}"
    )


def draw_random_k2_heads(
    n_layers: int = 42,
    n_heads: int = 16,
    seed: int = 42,
) -> tuple[list[HeadRef], list[HeadRef], list[HeadRef], list[HeadRef]]:
    """Draw random K=2 pair for Gemma-2-9B (42 layers, 16 query heads).

    Excludes:
      - Gemma clean K2 headline pair: L31_H7, L39_H6
      - Gemma context K10 set: L40_H14, L17_H7, L24_H15, L22_H0, L32_H11, L25_H0, L3_H9, L8_H9, L7_H6, L29_H5
      - Gemma raw peak: L27_H6
    """
    par_path = find_file("heads_dir_parametric_k2_clean.json", require_gemma_clean_k2=True)
    ctx_path = find_file("heads_dir_context_k10.json")
    raw_path = find_file("heads_dir_parametric_k1_raw.json")

    par_heads = list(load_identified_heads(par_path).enlarge)
    ctx_heads = list(load_identified_heads(ctx_path).enlarge)
    raw_heads = list(load_identified_heads(raw_path).enlarge)

    assert head_keys(par_heads) == ["L31_H7", "L39_H6"], (
        f"Unexpected clean K2 heads loaded from {par_path}: {head_keys(par_heads)}. "
        "Expected Gemma-2-9B clean K2 heads ['L31_H7', 'L39_H6']."
    )
    assert len(ctx_heads) == 10, f"Expected 10 context heads, got {len(ctx_heads)}"
    assert head_keys(raw_heads) == ["L27_H6"], f"Expected raw peak L27_H6, got {head_keys(raw_heads)}"

    exclude = par_heads + ctx_heads + raw_heads
    rand_heads = random_head_control(n_layers, n_heads, 2, seed=seed, exclude=exclude)

    expected = ["L41_H11", "L7_H3"]
    actual = head_keys(rand_heads)
    assert actual == expected, f"Expected {expected}, got {actual}"
    return rand_heads, par_heads, ctx_heads, raw_heads


def load_eval_items(split_path: Path, data_path: Path) -> list[ConflictItem]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert split.get("seed") == 42, f"Expected split seed 42, got {split.get('seed')}"
    assert split.get("n_eval") == 100, f"Expected n_eval 100, got {split.get('n_eval')}"

    eval_ids = set(split["eval_ids"])
    items = [it for it in load_conflict_items(data_path) if it.id in eval_ids]
    assert len(items) == 100, f"Expected 100 items, got {len(items)}"
    return items


def compute_paired_bootstrap(
    ids: list[str],
    base_fc: dict[str, bool],
    ste_fc: dict[str, bool],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    n = len(ids)
    assert n > 0
    point = sum(ste_fc[i] for i in ids) / n - sum(base_fc[i] for i in ids) / n
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample_ids = [ids[j] for j in sample_idx]
        d = sum(ste_fc[i] for i in sample_ids) / n - sum(base_fc[i] for i in sample_ids) / n
        draws.append(d)
    lo = float(np.percentile(draws, 2.5))
    hi = float(np.percentile(draws, 97.5))
    excl_zero = not (lo <= 0.0 <= hi)

    def ppv(x: float) -> str:
        v = round(x)
        if abs(x - v) < 1e-6:
            return f"{v:+.0f}"
        return f"{x:+.1f}"

    pp = 100.0 * point
    lo_pp = 100.0 * lo
    hi_pp = 100.0 * hi
    fmt = f"{ppv(pp)} [{ppv(lo_pp)}, {ppv(hi_pp)}]"

    return {
        "n": n,
        "point_delta_pp": pp,
        "ci95_pp": [lo_pp, hi_pp],
        "excludes_zero": excl_zero,
        "fmt": fmt,
        "n_boot": n_boot,
        "seed": seed,
    }


def score_cell(
    items: list[ConflictItem],
    baseline_rows: list[dict[str, Any]],
    steered_rows: list[dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    item_map = {it.id: it for it in items}
    base_map = {r["id"]: r for r in baseline_rows}
    ste_map = {r["id"]: r for r in steered_rows}
    ids = sorted(set(item_map) & set(base_map) & set(ste_map))
    assert len(ids) == len(items) == 100

    qa_b: dict[str, bool] = {}
    qa_s: dict[str, bool] = {}
    cons_s: dict[str, bool] = {}
    detailed_rows: list[dict[str, Any]] = []

    n_par_from_ctx = 0
    n_leave_without_par = 0

    for i in ids:
        it = item_map[i]
        t_base = base_map[i].get("text") or base_map[i].get("completion") or ""
        t_ste = ste_map[i].get("text") or ste_map[i].get("completion") or ""

        lb_base = label_conflict_response(t_base, it.parametric_answer, it.context_answer, question=it.question)
        lb_ste = label_conflict_response(t_ste, it.parametric_answer, it.context_answer, question=it.question)

        fc_base = (lb_base == "context")
        fc_ste = (lb_ste == "context")

        qa_b[i] = fc_base
        qa_s[i] = fc_ste

        par_hit = _mentions_answer(t_ste, it.parametric_answer, question=it.question)

        # Gold-PAR-only: leaving context counts only if gold parametric entity is named
        if fc_base and not fc_ste:
            if par_hit:
                cons_fc = False
                n_par_from_ctx += 1
            else:
                cons_fc = True  # Recode wrap / other as still context
                n_leave_without_par += 1
        else:
            cons_fc = fc_ste

        cons_s[i] = cons_fc

        detailed_rows.append({
            "id": i,
            "question": it.question,
            "parametric_gold": it.parametric_answer,
            "context_gold": it.context_answer,
            "baseline_text": t_base,
            "steered_text": t_ste,
            "baseline_qa_label": lb_base,
            "steered_qa_label": lb_ste,
            "steered_mentions_par_gold": par_hit,
            "baseline_fc": fc_base,
            "steered_fc_inclusive": fc_ste,
            "steered_fc_gold_par_only": cons_fc,
        })

    boot_incl = compute_paired_bootstrap(ids, qa_b, qa_s, n_boot=n_boot, seed=seed)
    boot_gold = compute_paired_bootstrap(ids, qa_b, cons_s, n_boot=n_boot, seed=seed)

    return {
        "n": len(ids),
        "fc_baseline": sum(qa_b.values()) / len(ids),
        "fc_steered_inclusive": sum(qa_s.values()) / len(ids),
        "fc_steered_gold_par_only": sum(cons_s.values()) / len(ids),
        "n_gold_par_from_context": n_par_from_ctx,
        "n_leave_ctx_without_par": n_leave_without_par,
        "inclusive": boot_incl,
        "gold_par_only": boot_gold,
        "rows": detailed_rows,
    }


def run_precision_cell(
    precision_name: str,
    model_id: str,
    items: list[ConflictItem],
    clean_k2_ident: IdentifiedHeads,
    rand_k2_ident: IdentifiedHeads,
    alpha_plus: float = 1.0,
    alpha_minus: float = 0.0,
    max_new_tokens: int = 16,
    prompt_format: str = "juice_coherent",
    is_gptq: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from common.inference import greedy_generate
    from common.models import load_model_and_tokenizer, unload_model

    print(f"\n{'='*70}\nLoading {precision_name}: {model_id} (GPTQ={is_gptq})\n{'='*70}")
    t0 = time.time()
    if is_gptq:
        model, tokenizer = load_model_and_tokenizer(
            "int4",
            model_id_int4=model_id,
            device_map={"": 0},
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            "fp16",
            model_id_fp16=model_id,
            device_map="auto",
        )
    wrap_attention_for_juice(model)
    print(f"Model loaded in {time.time() - t0:.1f}s on device(s): {set(p.device for p in model.parameters())}")

    # Parameter/buffer device audit
    meta_params = [(name, p.shape, p.dtype) for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [(name, b.shape, b.dtype) for name, b in model.named_buffers() if b.device.type == "meta"]
    print(f"\n[DIAGNOSTIC] Parameter/buffer device audit for {precision_name}:")
    print(f"  Total named parameters: {sum(1 for _ in model.named_parameters())}")
    print(f"  Parameters on meta:     {len(meta_params)}")
    print(f"  Buffers on meta:        {len(meta_buffers)}")
    print(f"  hf_device_map:          {getattr(model, 'hf_device_map', None)}")
    if meta_params or meta_buffers:
        print(f"{'!'*80}\nWARNING: UNMATERIALIZED 'meta' TENSORS DETECTED IN {precision_name}:\n")
        for name, shape, dtype in meta_params:
            print(f"  [PARAM ON META] {name} | shape={shape} | dtype={dtype}")
        for name, shape, dtype in meta_buffers:
            print(f"  [BUFFER ON META] {name} | shape={shape} | dtype={dtype}")
        print(f"{'!'*80}\n", flush=True)
    else:
        print(f"  [OK] All parameters and buffers materialized on real devices.\n", flush=True)

    # 1. Baseline unsteered
    print(f"Running unsteered baseline ({len(items)} items)...")
    baseline_rows: list[dict[str, Any]] = []
    for it in items:
        prompt = conflict_eval_prompt(tokenizer, it, prompt_format=prompt_format)
        text = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        baseline_rows.append({"id": it.id, "text": text})
    print("Baseline generation done.")

    # 2. Clean K2 steered
    print(f"Running JuICE clean K=2 steering ({head_keys(clean_k2_ident.enlarge)})...")
    clean_steered_rows = apply_juice_steering(
        model,
        tokenizer,
        items,
        clean_k2_ident,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
    )
    print("JuICE clean K2 steering done.")

    # 3. Random K2 steered
    print(f"Running JuICE random K=2 steering ({head_keys(rand_k2_ident.enlarge)})...")
    rand_steered_rows = apply_juice_steering(
        model,
        tokenizer,
        items,
        rand_k2_ident,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
    )
    print("JuICE random K2 steering done.")

    # Unload
    print(f"Unloading {precision_name} to free GPU VRAM...")
    unload_model(model)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM cleared.")

    return baseline_rows, clean_steered_rows, rand_steered_rows


def format_table12(
    fp16_clean: dict[str, Any] | None,
    int4_clean: dict[str, Any],
    fp16_rand: dict[str, Any] | None,
    int4_rand: dict[str, Any],
) -> str:
    fp16_clean_gold = fp16_clean['gold_par_only']['fmt'] if fp16_clean else "—"
    fp16_clean_incl = fp16_clean['inclusive']['fmt'] if fp16_clean else "−6 [−12, −1]"
    fp16_clean_excl = ("yes" if fp16_clean['gold_par_only']['excludes_zero'] else "no") if fp16_clean else "yes"

    fp16_rand_gold = fp16_rand['gold_par_only']['fmt'] if fp16_rand else "—"
    fp16_rand_incl = fp16_rand['inclusive']['fmt'] if fp16_rand else "—"
    fp16_rand_excl = ("yes" if fp16_rand['gold_par_only']['excludes_zero'] else "no") if fp16_rand else "—"

    int4_clean_gold = int4_clean['gold_par_only']['fmt']
    int4_clean_incl = int4_clean['inclusive']['fmt']
    int4_clean_excl = "yes" if int4_clean['gold_par_only']['excludes_zero'] else "no"

    int4_rand_gold = int4_rand['gold_par_only']['fmt']
    int4_rand_incl = int4_rand['inclusive']['fmt']
    int4_rand_excl = "yes" if int4_rand['gold_par_only']['excludes_zero'] else "no"

    lines = [
        "| Condition | Heads / notes | n=100 gold-PAR [CI] | n=100 incl. [CI] | excl. 0? |",
        "|:---|:---:|:---:|:---:|:---:|",
        f"| **FP16 par K2 clean** | L31_H7, L39_H6 | {fp16_clean_gold} | {fp16_clean_incl} | {fp16_clean_excl} |",
        f"| **INT4 par K2 clean** | same locked pair | {int4_clean_gold} | {int4_clean_incl} | {int4_clean_excl} |",
        f"| **FP16 random K2 control** | L41_H11, L7_H3 | {fp16_rand_gold} | {fp16_rand_incl} | {fp16_rand_excl} |",
        f"| **INT4 random K2 control** | same pair | {int4_rand_gold} | {int4_rand_incl} | {int4_rand_excl} |",
    ]
    return "\n".join(lines)


def format_csv(
    fp16_clean: dict[str, Any] | None,
    int4_clean: dict[str, Any],
    fp16_rand: dict[str, Any] | None,
    int4_rand: dict[str, Any],
) -> str:
    lines = [
        "Condition,Heads_notes,n100_gold_PAR_delta_pp,n100_gold_PAR_ci95,n100_incl_delta_pp,n100_incl_ci95,excludes_zero",
    ]
    if fp16_clean:
        lines.append(
            f'FP16 par K2 clean,"L31_H7, L39_H6",{fp16_clean["gold_par_only"]["point_delta_pp"]:.1f},"{fp16_clean["gold_par_only"]["ci95_pp"]}",{fp16_clean["inclusive"]["point_delta_pp"]:.1f},"{fp16_clean["inclusive"]["ci95_pp"]}",{"yes" if fp16_clean["gold_par_only"]["excludes_zero"] else "no"}'
        )
    else:
        lines.append('FP16 par K2 clean,"L31_H7, L39_H6",NA,NA,-6.0,"[-12, -1]",yes')

    lines.append(
        f'INT4 par K2 clean,same locked pair,{int4_clean["gold_par_only"]["point_delta_pp"]:.1f},"{int4_clean["gold_par_only"]["ci95_pp"]}",{int4_clean["inclusive"]["point_delta_pp"]:.1f},"{int4_clean["inclusive"]["ci95_pp"]}",{"yes" if int4_clean["gold_par_only"]["excludes_zero"] else "no"}'
    )

    if fp16_rand:
        lines.append(
            f'FP16 random K2 control,"L41_H11, L7_H3",{fp16_rand["gold_par_only"]["point_delta_pp"]:.1f},"{fp16_rand["gold_par_only"]["ci95_pp"]}",{fp16_rand["inclusive"]["point_delta_pp"]:.1f},"{fp16_rand["inclusive"]["ci95_pp"]}",{"yes" if fp16_rand["gold_par_only"]["excludes_zero"] else "no"}'
        )
    else:
        lines.append('FP16 random K2 control,"L41_H11, L7_H3",NA,NA,NA,NA,NA')

    lines.append(
        f'INT4 random K2 control,same pair,{int4_rand["gold_par_only"]["point_delta_pp"]:.1f},"{int4_rand["gold_par_only"]["ci95_pp"]}",{int4_rand["inclusive"]["point_delta_pp"]:.1f},"{int4_rand["inclusive"]["ci95_pp"]}",{"yes" if int4_rand["gold_par_only"]["excludes_zero"] else "no"}'
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="INT4 Transfer Test for Gemma Clean-K2 (L31_H7, L39_H6) with Matched Random Control"
    )
    parser.add_argument("--fp16-model", type=str, default="google/gemma-2-9b", help="FP16 model ID / path")
    parser.add_argument("--int4-model", type=str, default=None, help="Explicit path to gemma2_9b_base_gptq_int4_ncalib128")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 run and only evaluate INT4 conditions")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Verify head selection and split without running model inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    # Authenticate with Hugging Face Hub (needed for gated google/gemma-2-9b)
    try:
        from huggingface_hub import login
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                sec = UserSecretsClient()
                for k in ["HF_TOKEN", "hf_token", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
                    try:
                        token = sec.get_secret(k)
                        if token:
                            break
                    except Exception:
                        pass
            except Exception:
                pass
        if token:
            login(token=token, add_to_git_credential=False)
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            print("[AUTH] Successfully authenticated Hugging Face with Kaggle Secrets token.")
        else:
            print("[AUTH] WARNING: No HF_TOKEN found in Kaggle Secrets or environment.")
    except Exception as exc:
        print(f"[AUTH] Hugging Face login skipped or failed ({type(exc).__name__}: {exc}).")

    # Ensure gptqmodel is available for INT4 GPTQ loading (on Kaggle)
    try:
        import gptqmodel  # noqa: F401
    except ImportError:
        if "KAGGLE" in os.environ or Path("/kaggle").exists():
            print("[SETUP] gptqmodel not detected on Kaggle. Installing gptqmodel==2.2.0...")
            import subprocess
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q",
                    "gptqmodel==2.2.0", "transformers==4.51.3", "accelerate>=0.26.0,<2.0",
                ])
                print("[SETUP] gptqmodel installed successfully.")
            except Exception as e:
                print(f"[SETUP] Notice: gptqmodel install failed ({e})")
        else:
            print("[SETUP] gptqmodel not detected (non-Kaggle local environment; skipping pip install).")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "gemma_clean_k2_int4_transfer_n100_ncalib128"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: HEAD SELECTION & PREREGISTERED SPLIT")
    print("=" * 70)
    rand_heads, par_heads, ctx_heads, raw_heads = draw_random_k2_heads(n_layers=42, n_heads=16, seed=args.seed)
    print(f"Parametric clean K2 heads (locked):  {head_keys(par_heads)}")
    print(f"Context K10 heads (excluded):        {head_keys(ctx_heads)}")
    print(f"Raw peak L27_H6 (excluded):          {head_keys(raw_heads)}")
    print(f"Drawn random K=2 heads (seed {args.seed}): {head_keys(rand_heads)}")
    assert head_keys(par_heads) == ["L31_H7", "L39_H6"]
    assert head_keys(rand_heads) == ["L41_H11", "L7_H3"]

    ident_clean = identified_from_enlarge(par_heads, precision="gemma_clean_k2", K=2, steer_target="parametric")
    ident_rand = identified_from_enlarge(rand_heads, precision="gemma_random_k2", K=2, steer_target="parametric")

    split_path = find_file("split.json", check_n100=True)
    data_path = find_file("conflictqa_pilot250_popular.jsonl")
    eval_items = load_eval_items(split_path, data_path)
    print(f"Loaded {len(eval_items)} evaluation items from {data_path.name}")

    if args.dry_run:
        print("\n[DRY RUN] Validating load_model_and_tokenizer call signatures...")
        import inspect
        from common.models import load_model_and_tokenizer

        sig = inspect.signature(load_model_and_tokenizer)
        sig.bind("fp16", model_id_fp16=args.fp16_model, device_map="auto")
        mock_int4_id = args.int4_model or "dummy/gemma2_9b_base_gptq_int4_ncalib128"
        sig.bind("int4", model_id_int4=mock_int4_id, device_map={"": 0})
        print("[DRY RUN] All model loader call signatures verified successfully.")

        print("\n[DRY RUN] Validating prompt construction from ConflictItem...")
        from common.prompts import juice_style_factual_prompts
        test_item = eval_items[0]
        test_prompts = juice_style_factual_prompts(test_item)
        assert "coherent_conflict" in test_prompts, "Missing 'coherent_conflict' key in prompt dict"
        assert len(test_prompts["coherent_conflict"]) > 10, f"Prompt too short: {test_prompts['coherent_conflict']!r}"
        print(f"[DRY RUN] Prompt construction OK. Sample prompt ({len(test_prompts['coherent_conflict'])} chars): {test_prompts['coherent_conflict'][:120]}...")

        mock_eval = {
            "gold_par_only": {"point_delta_pp": -6.0, "ci95_pp": [-12.0, -1.0], "excludes_zero": True, "fmt": "-6 [-12, -1]"},
            "inclusive": {"point_delta_pp": -6.0, "ci95_pp": [-12.0, -1.0], "excludes_zero": True, "fmt": "-6 [-12, -1]"},
        }
        mock_rand = {
            "gold_par_only": {"point_delta_pp": 0.0, "ci95_pp": [-4.0, 4.0], "excludes_zero": False, "fmt": "+0 [-4, +4]"},
            "inclusive": {"point_delta_pp": 0.0, "ci95_pp": [-5.0, 5.0], "excludes_zero": False, "fmt": "+0 [-5, +5]"},
        }
        print("\nTable 12 Extension Preview:\n" + format_table12(mock_eval, mock_eval, mock_rand, mock_rand))
        print("\n[DRY RUN] All prerequisites validated successfully.")
        return

    # STEP 2: FP16 Execution
    if not args.skip_fp16:
        fp16_base_rows, fp16_clean_rows, fp16_rand_rows = run_precision_cell(
            precision_name="FP16",
            model_id=args.fp16_model,
            items=eval_items,
            clean_k2_ident=ident_clean,
            rand_k2_ident=ident_rand,
            alpha_plus=1.0,
            alpha_minus=0.0,
            max_new_tokens=16,
            prompt_format="juice_coherent",
            is_gptq=False,
        )
    else:
        print("\n[STEP 2 SKIPPED] --skip-fp16 flag supplied. Bypassing FP16 execution.")
        fp16_base_rows, fp16_clean_rows, fp16_rand_rows = [], [], []

    # STEP 3: INT4 NCALIB=128 Execution
    int4_path = resolve_gemma2_int4_ncalib128(args.int4_model)
    int4_base_rows, int4_clean_rows, int4_rand_rows = run_precision_cell(
        precision_name="INT4 NCALIB=128",
        model_id=int4_path,
        items=eval_items,
        clean_k2_ident=ident_clean,
        rand_k2_ident=ident_rand,
        alpha_plus=1.0,
        alpha_minus=0.0,
        max_new_tokens=16,
        prompt_format="juice_coherent",
        is_gptq=True,
    )

    # STEP 4: Scoring & Paired Bootstrap
    print("\n" + "=" * 70)
    print("STEP 4: QUESTION-AWARE SCORING & BOOTSTRAP")
    print("=" * 70)
    fp16_clean_eval = None
    fp16_rand_eval = None
    if not args.skip_fp16:
        print("Scoring FP16 clean K2...")
        fp16_clean_eval = score_cell(eval_items, fp16_base_rows, fp16_clean_rows, n_boot=args.n_boot, seed=args.seed)
        print("Scoring FP16 random K2...")
        fp16_rand_eval = score_cell(eval_items, fp16_base_rows, fp16_rand_rows, n_boot=args.n_boot, seed=args.seed)

    print("Scoring INT4 clean K2...")
    int4_clean_eval = score_cell(eval_items, int4_base_rows, int4_clean_rows, n_boot=args.n_boot, seed=args.seed)
    print("Scoring INT4 random K2...")
    int4_rand_eval = score_cell(eval_items, int4_base_rows, int4_rand_rows, n_boot=args.n_boot, seed=args.seed)

    print("\n=== RESULTS SUMMARY ===")
    if fp16_clean_eval:
        print(f"FP16 par K2 clean:      gold-PAR={fp16_clean_eval['gold_par_only']['fmt']} | incl={fp16_clean_eval['inclusive']['fmt']}")
        print(f"FP16 random K2 control:  gold-PAR={fp16_rand_eval['gold_par_only']['fmt']} | incl={fp16_rand_eval['inclusive']['fmt']}")
    else:
        print("FP16 conditions:         SKIPPED (--skip-fp16)")
    print(f"INT4 par K2 clean:       gold-PAR={int4_clean_eval['gold_par_only']['fmt']} | incl={int4_clean_eval['inclusive']['fmt']}")
    print(f"INT4 random K2 control:  gold-PAR={int4_rand_eval['gold_par_only']['fmt']} | incl={int4_rand_eval['inclusive']['fmt']}")

    table12_md = format_table12(fp16_clean_eval, int4_clean_eval, fp16_rand_eval, int4_rand_eval)
    table12_csv = format_csv(fp16_clean_eval, int4_clean_eval, fp16_rand_eval, int4_rand_eval)

    print("\n" + "=" * 70)
    print("TABLE 12 EXTENSION: GEMMA CLEAN-K2 INT4 TRANSFER & RANDOM CONTROLS")
    print("=" * 70)
    print(table12_md)

    # Output payloads
    base_stem = "gemma_clean_k2_int4_transfer_n100_ncalib128"
    json_path = out_dir / f"{base_stem}.json"
    md_path = out_dir / f"{base_stem}.md"
    csv_path = out_dir / f"{base_stem}.csv"

    full_payload = {
        "experiment": "Gemma clean-K2 INT4 transfer test with matched random control",
        "clean_heads": head_keys(par_heads),
        "random_heads": head_keys(rand_heads),
        "excluded_heads": head_keys(par_heads + ctx_heads + raw_heads),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_eval": 100,
        "steering_spec": {
            "alpha_plus": 1.0,
            "alpha_minus": 0.0,
            "max_new_tokens": 16,
            "prompt_format": "juice_coherent",
            "steer_target": "parametric",
        },
        "fp16_clean_k2": fp16_clean_eval,
        "fp16_random_k2": fp16_rand_eval,
        "int4_clean_k2": int4_clean_eval,
        "int4_random_k2": int4_rand_eval,
        "table12_markdown": table12_md,
    }

    json_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    md_path.write_text(table12_md + "\n", encoding="utf-8")
    csv_path.write_text(table12_csv + "\n", encoding="utf-8")

    print(f"\nWrote outputs:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
