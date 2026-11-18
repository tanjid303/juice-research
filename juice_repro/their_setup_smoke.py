#!/usr/bin/env python3
"""
Week 0 JuICE environment sanity check (Fix 2 parallel track).

Clones nothing automatically — expects external/JUICE after manual clone.
Swaps their default large model for Llama-3.2-1B-Instruct so nnsight hooks
run on 4GB VRAM.

Usage:
  git clone https://github.com/GaotangLi/JUICE external/JUICE
  python juice_repro/their_setup_smoke.py --n_items 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
JUICE_DIR = ROOT / "external" / "JUICE"
SMOKE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PINNED_NNSIGHT = "0.3.6"
PINNED_TRANSFORMERS = "4.48.0"


def check_imports() -> None:
    import nnsight  # noqa: F401
    import transformers

    nn_version = getattr(nnsight, "__version__", "unknown")
    if nn_version == "unknown":
        try:
            from importlib.metadata import version as pkg_version

            nn_version = pkg_version("nnsight")
        except Exception:
            nn_version = "unknown"

    print(f"nnsight={nn_version} (expected {PINNED_NNSIGHT})")
    print(f"transformers={transformers.__version__} (expected {PINNED_TRANSFORMERS})")
    if nn_version != "unknown" and nn_version != PINNED_NNSIGHT:
        print("WARNING: nnsight version differs from JUICE pin; match conda_env/requirement.txt")
    elif nn_version == "unknown":
        print("WARNING: could not determine nnsight version; continuing smoke test")


def check_juice_clone() -> None:
    if not JUICE_DIR.exists():
        print(f"JUICE repo not found at {JUICE_DIR}")
        print("Clone manually:")
        print("  git clone https://github.com/GaotangLi/JUICE external/JUICE")
        sys.exit(1)
    req = JUICE_DIR / "conda_env" / "requirement.txt"
    if req.exists():
        text = req.read_text(encoding="utf-8")
        if PINNED_NNSIGHT not in text:
            print(f"WARNING: {req} may not pin nnsight=={PINNED_NNSIGHT}")


def nnsight_trace_smoke(model_id: str, n_items: int) -> None:
    import torch
    from nnsight import LanguageModel
    from transformers import AutoTokenizer

    from common.conflictqa import load_conflict_items
    from common.prompts import apply_chat_template

    data_path = ROOT / "data" / "conflictqa_smoke.jsonl"
    items = load_conflict_items(data_path, n_items=n_items)
    item = items[0]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt = apply_chat_template(tok, item, "context_conflicts")

    print(f"Tracing {model_id} on 1 prompt ({len(items)} items loaded for context)...")
    lm = LanguageModel(model_id, device_map="auto", torch_dtype=torch.float16)
    n_layers = lm.config.num_hidden_layers
    n_heads = lm.config.num_attention_heads
    head_dim = lm.config.hidden_size // n_heads

    with lm.trace(prompt, scan=False, validate=False):
        # Minimal end-to-end hook: last-layer residual stream (confirms nnsight works).
        # Head-slice intervention for JuICE lives in common/juice.py (torch hooks).
        layer = n_layers - 1
        residual = lm.model.layers[layer].output[0]
        saved = residual[:, -1, :].save()

    print(
        f"Trace OK. L{layer} residual last-token shape: {tuple(saved.shape)} "
        f"(n_heads={n_heads}, head_dim={head_dim})"
    )
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("nnsight hook smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="JuICE repo + nnsight smoke (1B, 4GB-safe)")
    parser.add_argument("--model_id", default=SMOKE_MODEL)
    parser.add_argument("--n_items", type=int, default=5, help="Items to load from smoke JSONL")
    parser.add_argument("--skip_trace", action="store_true", help="Only check imports/clone")
    args = parser.parse_args()

    check_imports()
    check_juice_clone()

    if args.skip_trace:
        print("Skip trace requested; import/clone checks only.")
        return

    nnsight_trace_smoke(args.model_id, args.n_items)


if __name__ == "__main__":
    main()
