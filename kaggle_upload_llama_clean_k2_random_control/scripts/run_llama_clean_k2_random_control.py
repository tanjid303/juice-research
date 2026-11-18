"""Matched random-K2 control for Llama clean-K2 headline finding.

Runs random K=2 attention-head control (seed 42, excluding clean K2 and ctx K10)
on both FP16 (meta-llama/Llama-3.1-8B) and canonical INT4 (llama31_8b_base_gptq_int4_ncalib128)
under dual-run JuICE steering (alpha+=1, alpha-=0, greedy decoding, max_new_tokens=16)
on the exact seed-42 n=100 evaluation set.

Scores via question-aware matcher and reports:
  - n=100 gold-PAR-only delta [95% CI]
  - n=100 inclusive delta [95% CI]
  - excludes 0?

Outputs:
  llama_clean_k2_random_control_n100_ncalib128.json
  llama_clean_k2_random_control_n100_ncalib128.md
  llama_clean_k2_random_control_n100_ncalib128.csv
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
)
from common.scoring import _mentions_answer, label_conflict_response


def find_file(name: str, check_n100: bool = False, require_llama_k2: bool = False) -> Path:
    # First check direct standard locations
    direct_candidates = [
        REPO / "directional_heads" / name,
        REPO / "data" / name,
        REPO / "juice_stage2_base_id" / "directional_heads" / name,
        REPO / "kaggle_upload_llama_clean_k2_random_control" / "directional_heads" / name,
        REPO / "kaggle_upload_llama_clean_k2_random_control" / "data" / name,
        REPO / "kaggle_upload_stage2_n100_headline" / "directional_heads" / name,
        REPO / "kaggle_upload_stage2_n100_headline" / "data" / name,
        Path("/kaggle/working") / name,
        Path("/kaggle/working") / "directional_heads" / name,
        Path("/kaggle/working") / "data" / name,
        Path("/kaggle/input") / name,
    ]
    for c in direct_candidates:
        if c.is_file():
            if "gemma" in str(c).lower() and "llama" not in str(c).lower():
                continue
            if check_n100 and name == "split.json":
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    if data.get("n_eval") == 100:
                        return c
                except Exception:
                    continue
            elif require_llama_k2 and name == "heads_dir_parametric_k2_clean.json":
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    h_keys = [f"L{h['layer']}_H{h['head']}" for h in data.get("enlarge", [])]
                    if h_keys == ["L20_H14", "L26_H15"]:
                        return c
                except Exception:
                    continue
            else:
                return c

    # Fallback search in working or input
    search_roots = [Path("/kaggle/working"), Path("/kaggle/input"), REPO]
    hits: list[Path] = []
    for root in search_roots:
        if root.exists():
            for p in root.rglob(name):
                if p.is_file():
                    # Reject obvious Gemma files when looking for Llama assets
                    if "gemma" in str(p).lower() and "llama" not in str(p).lower():
                        continue
                    if check_n100 and name == "split.json":
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            if data.get("n_eval") == 100:
                                hits.append(p)
                        except Exception:
                            pass
                    elif require_llama_k2 and name == "heads_dir_parametric_k2_clean.json":
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            h_keys = [f"L{h['layer']}_H{h['head']}" for h in data.get("enlarge", [])]
                            if h_keys == ["L20_H14", "L26_H15"]:
                                hits.append(p)
                        except Exception:
                            pass
                    else:
                        hits.append(p)
    if not hits:
        raise FileNotFoundError(f"Could not find required file: {name} (check_n100={check_n100}, require_llama_k2={require_llama_k2})")
    # Prioritize Llama-specific paths, then n100/stage2, then shorter paths
    hits.sort(key=lambda p: (
        0 if "llama" in str(p).lower() else 1,
        0 if "n100" in str(p) or "stage2" in str(p) else 1,
        len(str(p))
    ))
    return hits[0]


def resolve_int4_ncalib128(explicit_path: str | None = None) -> str:
    from common.models import _looks_like_gptq_dir

    named = "llama31_8b_base_gptq_int4_ncalib128"
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
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
        if _looks_like_gptq_dir(path):
            print(f"Resolved canonical INT4 NCALIB=128 GPTQ directory -> {path}")
            return str(path)
    raise FileNotFoundError(
        f"Canonical N_CALIB=128 INT4 checkpoint '{named}' not found. "
        f"Searched: {list(seen)[:10]}"
    )


def draw_random_k2_heads(
    n_layers: int = 32,
    n_heads: int = 32,
    seed: int = 42,
) -> tuple[list[HeadRef], list[HeadRef], list[HeadRef]]:
    """Draw random K=2 pair excluding clean K2 and context K10."""
    par_path = find_file("heads_dir_parametric_k2_clean.json", require_llama_k2=True)
    ctx_path = find_file("heads_dir_context_k10.json")

    par_heads = list(load_identified_heads(par_path).enlarge)
    ctx_heads = list(load_identified_heads(ctx_path).enlarge)

    assert head_keys(par_heads) == ["L20_H14", "L26_H15"], (
        f"Unexpected clean K2 heads loaded from {par_path}: {head_keys(par_heads)}. "
        "Expected Llama's clean K2 heads ['L20_H14', 'L26_H15']."
    )
    assert len(ctx_heads) == 10, f"Expected 10 context heads, got {len(ctx_heads)}"

    exclude = par_heads + ctx_heads
    rand_heads = random_head_control(n_layers, n_heads, 2, seed=seed, exclude=exclude)

    expected = ["L20_H24", "L3_H19"]
    actual = head_keys(rand_heads)
    assert actual == expected, f"Expected {expected}, got {actual}"
    return rand_heads, par_heads, ctx_heads


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
    ident_rand: IdentifiedHeads,
    alpha_plus: float = 1.0,
    alpha_minus: float = 0.0,
    max_new_tokens: int = 16,
    prompt_format: str = "juice_coherent",
    is_gptq: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    if is_gptq:
        from common.juice import wrap_llama_attention_for_juice
        wrap_llama_attention_for_juice(model)
    print(f"Model loaded in {time.time() - t0:.1f}s on device(s): {set(p.device for p in model.parameters())}")

    # Diagnostic scan: iterate model.named_parameters() and print any on 'meta' device
    meta_params = [(name, p.shape, p.dtype) for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [(name, b.shape, b.dtype) for name, b in model.named_buffers() if b.device.type == "meta"]
    print(f"\n[DIAGNOSTIC] Parameter/buffer device audit for {precision_name}:")
    print(f"  Total named parameters: {sum(1 for _ in model.named_parameters())}")
    print(f"  Parameters on meta:     {len(meta_params)}")
    print(f"  Buffers on meta:        {len(meta_buffers)}")
    print(f"  hf_device_map:          {getattr(model, 'hf_device_map', None)}")
    if meta_params or meta_buffers:
        print(f"{'!'*80}")
        print(f"WARNING: UNMATERIALIZED 'meta' TENSORS DETECTED IN {precision_name} MODEL:")
        for name, shape, dtype in meta_params:
            print(f"  [PARAM ON META] {name} | shape={shape} | dtype={dtype}")
        for name, shape, dtype in meta_buffers:
            print(f"  [BUFFER ON META] {name} | shape={shape} | dtype={dtype}")
        print(f"{'!'*80}\n", flush=True)
    else:
        print(f"  [OK] All parameters and buffers materialized on real devices (no meta tensors).\n", flush=True)

    # 1. Baseline unsteered
    print(f"Running unsteered baseline ({len(items)} items)...")
    baseline_rows: list[dict[str, Any]] = []
    for it in items:
        prompt = conflict_eval_prompt(tokenizer, it, prompt_format=prompt_format)
        text = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        baseline_rows.append({"id": it.id, "text": text})
    print("Baseline generation done.")

    # 2. Random K2 steered
    print(f"Running JuICE random K=2 steering ({head_keys(ident_rand.enlarge)})...")
    steered_rows = apply_juice_steering(
        model,
        tokenizer,
        items,
        ident_rand,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
    )
    print("JuICE steering generation done.")

    # Unload
    print(f"Unloading {precision_name} to free GPU VRAM...")
    unload_model(model)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM cleared.")

    return baseline_rows, steered_rows


def format_table8(fp16_eval: dict[str, Any], int4_eval: dict[str, Any]) -> str:
    lines = [
        "| Condition | n=100 gold-PAR | n=100 incl. | excl. 0? |",
        "|:---|:---:|:---:|:---:|",
        "| **FP16 clean K2** (headline locked pair) | −8 [−14, −3] | −15 [−22, −7] | yes |",
        f"| **FP16 random K2** (matched control: L20_H24+L3_H19) | {fp16_eval['gold_par_only']['fmt']} | {fp16_eval['inclusive']['fmt']} | {'yes' if fp16_eval['gold_par_only']['excludes_zero'] else 'no'} |",
        "| **INT4 clean K2 transfer** (headline locked pair) | −7 [−12, −2] | −12 [−19, −6] | yes |",
        f"| **INT4 random K2** (matched control: L20_H24+L3_H19) | {int4_eval['gold_par_only']['fmt']} | {int4_eval['inclusive']['fmt']} | {'yes' if int4_eval['gold_par_only']['excludes_zero'] else 'no'} |",
    ]
    return "\n".join(lines)


def format_csv(fp16_eval: dict[str, Any], int4_eval: dict[str, Any]) -> str:
    lines = [
        "Condition,n100_gold_PAR_delta_pp,n100_gold_PAR_ci95,n100_incl_delta_pp,n100_incl_ci95,excludes_zero",
        'FP16 clean K2 (headline),-8,"[-14, -3]",-15,"[-22, -7]",yes',
        f'FP16 random K2 (control),{fp16_eval["gold_par_only"]["point_delta_pp"]:.1f},"{fp16_eval["gold_par_only"]["ci95_pp"]}",{fp16_eval["inclusive"]["point_delta_pp"]:.1f},"{fp16_eval["inclusive"]["ci95_pp"]}",{"yes" if fp16_eval["gold_par_only"]["excludes_zero"] else "no"}',
        'INT4 clean K2 transfer (headline),-7,"[-12, -2]",-12,"[-19, -6]",yes',
        f'INT4 random K2 (control),{int4_eval["gold_par_only"]["point_delta_pp"]:.1f},"{int4_eval["gold_par_only"]["ci95_pp"]}",{int4_eval["inclusive"]["point_delta_pp"]:.1f},"{int4_eval["inclusive"]["ci95_pp"]}",{"yes" if int4_eval["gold_par_only"]["excludes_zero"] else "no"}',
    ]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Matched Random-K2 Control for Llama Clean-K2 (FP16 & INT4 NCALIB=128)")
    parser.add_argument("--fp16-model", type=str, default="meta-llama/Llama-3.1-8B", help="FP16 model ID / path")
    parser.add_argument("--int4-model", type=str, default=None, help="Explicit path to llama31_8b_base_gptq_int4_ncalib128")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 run and only evaluate INT4 control")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Verify head selection and split without running model inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    # Authenticate with Hugging Face Hub (needed for gated meta-llama models)
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

    # Ensure gptqmodel is available for INT4 GPTQ loading (needed on Kaggle Py3.12)
    try:
        import gptqmodel  # noqa: F401
    except ImportError:
        print("[SETUP] gptqmodel not detected. Installing gptqmodel==2.2.0...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "gptqmodel==2.2.0", "transformers==4.51.3", "accelerate>=0.26.0,<2.0",
        ])
        print("[SETUP] gptqmodel installed successfully.")

    out_dir = Path(args.output_dir) if args.output_dir else (Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "llama_clean_k2_random_control")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: HEAD SELECTION & PREREGISTERED SPLIT")
    print("=" * 70)
    rand_heads, par_heads, ctx_heads = draw_random_k2_heads(n_layers=32, n_heads=32, seed=args.seed)
    print(f"Parametric clean K2 heads (excluded): {head_keys(par_heads)}")
    print(f"Context K10 heads (excluded):          {head_keys(ctx_heads)}")
    print(f"Drawn random K=2 heads (seed {args.seed}):   {head_keys(rand_heads)}")
    assert head_keys(rand_heads) == ["L20_H24", "L3_H19"]

    ident_rand = identified_from_enlarge(rand_heads, precision="llama_random_k2", K=2, steer_target="parametric")

    split_path = find_file("split.json", check_n100=True)
    data_path = find_file("conflictqa_pilot250_popular.jsonl")
    eval_items = load_eval_items(split_path, data_path)
    print(f"Loaded {len(eval_items)} evaluation items from {data_path.name}")

    if args.dry_run:
        print("\n[DRY RUN] Validating load_model_and_tokenizer call signatures...")
        import inspect
        from common.models import load_model_and_tokenizer

        sig = inspect.signature(load_model_and_tokenizer)
        # Test FP16 binding matching run_precision_cell
        sig.bind("fp16", model_id_fp16=args.fp16_model, device_map="auto")
        # Test INT4 binding matching run_precision_cell
        mock_int4_id = args.int4_model or "dummy/llama31_8b_base_gptq_int4_ncalib128"
        sig.bind("int4", model_id_int4=mock_int4_id, device_map={"": 0})
        print("[DRY RUN] All model loader call signatures verified successfully.")

        # Smoke-test prompt construction from a real ConflictItem (catches stale API bugs)
        print("\n[DRY RUN] Validating prompt construction from ConflictItem...")
        from common.prompts import juice_style_factual_prompts
        test_item = eval_items[0]
        test_prompts = juice_style_factual_prompts(test_item)
        assert "coherent_conflict" in test_prompts, "Missing 'coherent_conflict' key in prompt dict"
        assert len(test_prompts["coherent_conflict"]) > 10, f"Prompt too short: {test_prompts['coherent_conflict']!r}"
        print(f"[DRY RUN] Prompt construction OK. Sample prompt ({len(test_prompts['coherent_conflict'])} chars): {test_prompts['coherent_conflict'][:120]}...")

        print("\n[DRY RUN] All prerequisites validated successfully. Skipping heavy model inference.")
        mock_fp16 = {
            "gold_par_only": {"point_delta_pp": 0.0, "ci95_pp": [-4.0, 4.0], "excludes_zero": False, "fmt": "+0 [-4, +4]"},
            "inclusive": {"point_delta_pp": 0.0, "ci95_pp": [-5.0, 5.0], "excludes_zero": False, "fmt": "+0 [-5, +5]"},
        }
        mock_int4 = {
            "gold_par_only": {"point_delta_pp": 0.0, "ci95_pp": [-4.0, 4.0], "excludes_zero": False, "fmt": "+0 [-4, +4]"},
            "inclusive": {"point_delta_pp": 0.0, "ci95_pp": [-5.0, 5.0], "excludes_zero": False, "fmt": "+0 [-5, +5]"},
        }
        print("\nTable 8 Template Preview:\n" + format_table8(mock_fp16, mock_int4))
        return

    # STEP 2: FP16 Execution
    if not args.skip_fp16:
        fp16_base_rows, fp16_ste_rows = run_precision_cell(
            precision_name="FP16",
            model_id=args.fp16_model,
            items=eval_items,
            ident_rand=ident_rand,
            alpha_plus=1.0,
            alpha_minus=0.0,
            max_new_tokens=16,
            prompt_format="juice_coherent",
            is_gptq=False,
        )
    else:
        print("\n[STEP 2 SKIPPED] --skip-fp16 flag supplied. Bypassing FP16 execution.")
        fp16_base_rows, fp16_ste_rows = [], []

    # STEP 3: INT4 NCALIB=128 Execution
    int4_path = resolve_int4_ncalib128(args.int4_model)
    int4_base_rows, int4_ste_rows = run_precision_cell(
        precision_name="INT4 NCALIB=128",
        model_id=int4_path,
        items=eval_items,
        ident_rand=ident_rand,
        alpha_plus=1.0,
        alpha_minus=0.0,
        max_new_tokens=16,
        prompt_format="juice_coherent",
        is_gptq=True,
    )

    # STEP 4: Question-Aware Scoring & Bootstrap
    print("\n" + "=" * 70)
    print("STEP 4: QUESTION-AWARE SCORING & BOOTSTRAP")
    print("=" * 70)
    if not args.skip_fp16:
        fp16_results = score_cell(eval_items, fp16_base_rows, fp16_ste_rows, n_boot=args.n_boot, seed=args.seed)
    else:
        fp16_results = {
            "gold_par_only": {"point_delta_pp": 0.0, "ci95_pp": [0.0, 0.0], "excludes_zero": False, "fmt": "SKIPPED"},
            "inclusive": {"point_delta_pp": 0.0, "ci95_pp": [0.0, 0.0], "excludes_zero": False, "fmt": "SKIPPED"},
        }
    int4_results = score_cell(eval_items, int4_base_rows, int4_ste_rows, n_boot=args.n_boot, seed=args.seed)

    print("\n=== RESULTS SUMMARY ===")
    if not args.skip_fp16:
        print(f"FP16 random K2:  gold-PAR={fp16_results['gold_par_only']['fmt']} | incl={fp16_results['inclusive']['fmt']}")
    else:
        print("FP16 random K2:  SKIPPED (--skip-fp16)")
    print(f"INT4 random K2:  gold-PAR={int4_results['gold_par_only']['fmt']} | incl={int4_results['inclusive']['fmt']}")

    table8_md = format_table8(fp16_results, int4_results)
    table8_csv = format_csv(fp16_results, int4_results)

    print("\n" + "=" * 70)
    print("TABLE 8 COMPARISON (with matched random-K2 controls)")
    print("=" * 70)
    print(table8_md)

    # Output payloads
    base_stem = "llama_clean_k2_random_control_n100_ncalib128"
    json_path = out_dir / f"{base_stem}.json"
    md_path = out_dir / f"{base_stem}.md"
    csv_path = out_dir / f"{base_stem}.csv"

    full_payload = {
        "experiment": "Llama clean-K2 matched random control",
        "random_heads": head_keys(rand_heads),
        "excluded_heads": head_keys(par_heads + ctx_heads),
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
        "fp16": {
            "model": args.fp16_model,
            "fc_baseline": fp16_results["fc_baseline"],
            "fc_steered_inclusive": fp16_results["fc_steered_inclusive"],
            "fc_steered_gold_par_only": fp16_results["fc_steered_gold_par_only"],
            "gold_par_only": fp16_results["gold_par_only"],
            "inclusive": fp16_results["inclusive"],
            "n_gold_par_from_context": fp16_results["n_gold_par_from_context"],
            "n_leave_ctx_without_par": fp16_results["n_leave_ctx_without_par"],
        },
        "int4": {
            "model": int4_path,
            "fc_baseline": int4_results["fc_baseline"],
            "fc_steered_inclusive": int4_results["fc_steered_inclusive"],
            "fc_steered_gold_par_only": int4_results["fc_steered_gold_par_only"],
            "gold_par_only": int4_results["gold_par_only"],
            "inclusive": int4_results["inclusive"],
            "n_gold_par_from_context": int4_results["n_gold_par_from_context"],
            "n_leave_ctx_without_par": int4_results["n_leave_ctx_without_par"],
        },
        "table8_markdown": table8_md,
    }

    json_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    md_path.write_text(table8_md + "\n", encoding="utf-8")
    csv_path.write_text(table8_csv + "\n", encoding="utf-8")

    print(f"\nWrote outputs:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
