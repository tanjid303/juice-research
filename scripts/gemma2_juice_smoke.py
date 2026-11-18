"""Gemma-2-9B JuICE port smoke tests.

Stages:
  dispatch  — no weights. Config routing + eager softcap math (CPU).
  fp16      — load google/gemma-2-9b eager FP16, wrap, one dual-run forward.
  gptq      — 1-layer / 8-sample GPTQ, greedy-decode a few prompts.

GPU is used only for fp16 forward and gptq. dispatch is always CPU.

  python scripts/gemma2_juice_smoke.py --stage dispatch
  python scripts/gemma2_juice_smoke.py --stage fp16
  python scripts/gemma2_juice_smoke.py --stage gptq
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "google/gemma-2-9b"
EXPECTED = {
    "head_dim": 256,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "attn_logit_softcapping": 50.0,
    "hidden_size": 3584,
    "num_hidden_layers": 42,
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def stage_dispatch() -> dict:
    import torch

    from common.juice_attn import (
        GetHeadsGemma2Attention,
        GetHeadsLlamaAttention,
        eager_attention_forward,
        juice_attn_class_for_config,
    )

    gemma_cfg = SimpleNamespace(
        model_type="gemma2",
        architectures=["Gemma2ForCausalLM"],
    )
    llama_cfg = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
    )
    assert juice_attn_class_for_config(gemma_cfg) is GetHeadsGemma2Attention
    assert juice_attn_class_for_config(llama_cfg) is GetHeadsLlamaAttention
    _log("dispatch: gemma2 -> GetHeadsGemma2Attention, llama -> GetHeadsLlamaAttention")

    # Softcap must squash scores (HF Gemma-2 eager path).
    class _M:
        num_key_value_groups = 1
        training = False

    b, h, s, d = 1, 2, 4, 8
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, s, d, generator=g)
    k = torch.randn(b, h, s, d, generator=g)
    v = torch.randn(b, h, s, d, generator=g)
    out_capped, w_capped = eager_attention_forward(
        _M(), q, k, v, None, scaling=1.0, softcap=50.0
    )
    out_plain, w_plain = eager_attention_forward(
        _M(), q, k, v, None, scaling=1.0, softcap=None
    )
    assert out_capped.shape == (b, s, h, d)
    assert not torch.allclose(w_capped, w_plain), "softcap should change attn weights"
    _log("dispatch: eager softcap changes attn weights (ok)")
    return {"ok": True, "stage": "dispatch"}


def _print_and_check_config(cfg) -> dict:
    got = {
        "model_type": getattr(cfg, "model_type", None),
        "head_dim": int(getattr(cfg, "head_dim", -1)),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "attn_logit_softcapping": float(cfg.attn_logit_softcapping),
        "query_pre_attn_scalar": int(getattr(cfg, "query_pre_attn_scalar", -1)),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "sliding_window": getattr(cfg, "sliding_window", None),
        "final_logit_softcapping": getattr(cfg, "final_logit_softcapping", None),
        "_attn_implementation": getattr(cfg, "_attn_implementation", None),
    }
    _log("config: " + json.dumps(got, indent=2, default=str))
    mismatches = []
    for k, exp in EXPECTED.items():
        actual = got[k]
        if actual != exp:
            mismatches.append(f"{k}: got {actual}, expected {exp}")
    if mismatches:
        raise AssertionError("config mismatch vs audit:\n  " + "\n  ".join(mismatches))
    _log("config matches audit (256 / 16 / 8 / 50.0 / hidden 3584 / 42 layers)")
    return got


def _gemma2_geometry_model(device: str):
    """1-layer Gemma-2 with 9B attention geometry. No 18GB weight download."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    _print_and_check_config(cfg)
    cfg.num_hidden_layers = 1
    cfg.vocab_size = 1024
    cfg.intermediate_size = 1024
    cfg._attn_implementation = "eager"
    _log(
        "geometry model: 1 layer, vocab=1024, intermediate=1024, "
        "keeping 9B head_dim/heads/hidden/softcap"
    )
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    model.eval()
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if device.startswith("cuda"):
        model = model.to(device=torch.device("cuda:0"), dtype=dtype)
    else:
        model = model.to(dtype=dtype)
    return model, cfg


def _tiny_tokenizer_ids(seq_len: int, vocab: int, device):
    import torch

    input_ids = torch.randint(1, max(2, vocab - 1), (1, seq_len), device=device)
    mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": mask}


def _load_fp16(device: str):
    """Full google/gemma-2-9b eager FP16. Needs T4 x2 (~18GB weights)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": 0} if torch.cuda.device_count() == 1 else "auto"
    else:
        kwargs["device_map"] = "cpu"
    _log(f"loading {MODEL_ID} dtype={dtype} device_map={kwargs['device_map']}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return model, tok


def _first_conflict_prompt() -> str:
    from common.conflictqa import load_conflict_items
    from common.prompts import juice_style_factual_prompts

    path = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
    items = load_conflict_items(path, n_items=1)
    return juice_style_factual_prompts(items[0])["coherent_conflict"]


def stage_fp16(*, device: str, full_weights: bool = False) -> dict:
    import torch

    from common.juice import _set_head_out, dual_run_logits, wrap_attention_for_juice
    from common.juice_attn import GetHeadsGemma2Attention
    from common.models import input_device

    t0 = time.time()
    if full_weights:
        model, tok = _load_fp16(device)
        cfg = _print_and_check_config(model.config)
        prompt = _first_conflict_prompt()
        _log(f"prompt chars={len(prompt)} head={prompt[:80]!r}...")
        dev = input_device(model)
        enc = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in enc.items()}
        seq_len = int(inputs["input_ids"].shape[1])
        n_layers_expect = EXPECTED["num_hidden_layers"]
    else:
        model, cfg_obj = _gemma2_geometry_model(device)
        cfg = {
            "head_dim": int(cfg_obj.head_dim),
            "num_attention_heads": int(cfg_obj.num_attention_heads),
            "query_pre_attn_scalar": int(cfg_obj.query_pre_attn_scalar),
            "attn_logit_softcapping": float(cfg_obj.attn_logit_softcapping),
            "hidden_size": int(cfg_obj.hidden_size),
        }
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        prompt = _first_conflict_prompt()
        _log(f"prompt chars={len(prompt)} head={prompt[:80]!r}...")
        dev = next(model.parameters()).device
        enc = tok(prompt, return_tensors="pt")
        vocab = int(model.config.vocab_size)
        input_ids = enc["input_ids"].clamp(min=1, max=vocab - 1).to(dev)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].to(dev),
        }
        seq_len = int(inputs["input_ids"].shape[1])
        n_layers_expect = 1
        _log("fp16 geometry clone (full 9B weights skipped: use --full-weights on T4)")

    wrap_attention_for_juice(model)
    n_wrapped = sum(
        1
        for layer in model.model.layers
        if isinstance(layer.self_attn, GetHeadsGemma2Attention)
    )
    assert n_wrapped == n_layers_expect, n_wrapped
    sample = model.model.layers[0].self_attn
    _log(
        f"wrap: {n_wrapped} layers; layer0 scaling={sample.scaling:.6f} "
        f"softcap={sample.attn_logit_softcapping} sliding_window={sample.sliding_window}"
    )
    expected_scale = float(cfg["query_pre_attn_scalar"]) ** -0.5
    assert abs(sample.scaling - expected_scale) < 1e-6, (sample.scaling, expected_scale)

    _log(f"seq_len={seq_len} device={dev}")
    _set_head_out(model, True)
    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)

    bad = []
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        n_h = len(attn.head_output)
        if n_h != EXPECTED["num_attention_heads"]:
            bad.append(f"L{li} n_heads={n_h}")
            continue
        shp = tuple(attn.head_output[0].shape)
        if shp[0] != 1 or shp[1] != seq_len or shp[2] != EXPECTED["hidden_size"]:
            bad.append(f"L{li} shape={shp}")
    _set_head_out(model, False)
    if bad:
        raise AssertionError("head_output check failed: " + "; ".join(bad[:8]))
    _log(
        f"head_output: all {n_wrapped} layers have "
        f"{EXPECTED['num_attention_heads']} heads, shape [1, {seq_len}, {EXPECTED['hidden_size']}]"
    )

    with torch.inference_mode():
        logits = dual_run_logits(model, inputs, [(0, 0, 1.0)])
    _log(f"dual_run_logits ok: shape={tuple(logits.shape)} argmax={int(logits.argmax())}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "ok": True,
        "stage": "fp16",
        "full_weights": full_weights,
        "seconds": round(time.time() - t0, 1),
        "seq_len": seq_len,
        "n_layers_wrapped": n_wrapped,
        "head_output_shape": [1, seq_len, EXPECTED["hidden_size"]],
        "n_heads": EXPECTED["num_attention_heads"],
    }


def _patch_gemma2_layer_tuple() -> None:
    """gptqmodel 2.2 + transformers 4.51: decoder layer must return a tuple."""
    try:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
    except ImportError:
        _log("no Gemma2DecoderLayer to patch")
        return
    import torch

    orig = Gemma2DecoderLayer.forward

    def _fwd(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if torch.is_tensor(out):
            return (out,)
        return out

    Gemma2DecoderLayer.forward = _fwd
    _log("Gemma2DecoderLayer.forward patched to always return a tuple")


def stage_gptq(*, out_dir: Path, full_weights: bool = False) -> dict:
    try:
        import gptqmodel  # noqa: F401
    except ImportError:
        _log("SKIP gptq: gptqmodel not installed (Windows wheel build needs torch in isolated env).")
        _log("Run this stage on Kaggle (gptqmodel==2.2, transformers 4.51.3, T4).")
        return {
            "ok": False,
            "stage": "gptq",
            "skipped": True,
            "reason": "gptqmodel not installed locally",
        }

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPTQ smoke needs CUDA")

    import gptqmodel
    from gptqmodel import GPTQModel

    try:
        from gptqmodel import QuantizeConfig
    except ImportError:
        from gptqmodel import GPTQConfig as QuantizeConfig

    _patch_gemma2_layer_tuple()

    n_calib = 8
    max_seq = 256
    out_dir.mkdir(parents=True, exist_ok=True)

    # 8 short calib strings — skip C4 download for the tiny smoke.
    calib_texts = [
        "The capital of France is Paris, a city on the Seine.",
        "Gemma is a family of open language models released by Google.",
        "Question: Who wrote Pride and Prejudice?\nAnswer: Jane Austen.",
        "Sliding window attention alternates with global attention layers.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Python is a programming language used in scientific computing.",
        "A transformer decoder layer contains self-attention and an MLP.",
        "Calibration text for a one-layer GPTQ smoke test of Gemma-2.",
    ]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    calib_examples = []
    for text in calib_texts[:n_calib]:
        tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq)
        calib_examples.append(
            {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
        )

    # Quantize ALL layers. Skipping layers does not save the ~18GB CPU load
    # (that is what SIGKILL'd the subprocess smoke on Kaggle T4).
    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        damp_percent=0.01,
        parallel_packing=False,
        device="cuda:0",
    )
    _log(f"gptqmodel {gptqmodel.__version__} n_calib={n_calib} (all layers, no dynamic skip)")

    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()
    load_kw = {"trust_remote_code": True, "torch_dtype": torch.float16}
    try:
        model = GPTQModel.load(MODEL_ID, quant_config, **load_kw)
    except TypeError:
        model = GPTQModel.load(MODEL_ID, quant_config, trust_remote_code=True)
    model.quantize(
        calib_examples,
        batch_size=1,
        calibration_enable_gpu_cache=False,
        auto_gc=True,
        buffered_fwd=False,
    )
    quant_s = time.time() - t0
    _log(f"quantize done in {quant_s / 60:.1f} min")
    model.save(str(out_dir))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    try:
        from gptqmodel import BACKEND

        qmodel = GPTQModel.load(str(out_dir), backend=BACKEND.TORCH)
        _log("loaded BACKEND.TORCH")
    except Exception as exc:
        _log(f"BACKEND.TORCH failed ({exc}); default load")
        qmodel = GPTQModel.load(str(out_dir))

    prompts = [
        "The capital of France is",
        "Question: What is 2 + 2?\nAnswer:",
        "Once upon a time",
    ]
    generations = []
    garbage = []
    for p in prompts:
        with torch.inference_mode():
            result = qmodel.generate(p, max_new_tokens=16)
        text = result[0] if isinstance(result, (list, tuple)) else str(result)
        generations.append({"prompt": p, "out": text})
        _log(f"GEN {p!r} -> {text!r}")
        body = text[len(p) :] if text.startswith(p) else text
        compact = "".join(body.split())
        if len(compact) < 3 or compact == compact[0] * len(compact):
            garbage.append(p)

    del qmodel
    gc.collect()
    torch.cuda.empty_cache()
    if garbage:
        raise AssertionError(f"garbage generations for prompts: {garbage}")
    _log("GPTQ smoke: generations look non-garbage")
    meta = {
        "ok": True,
        "stage": "gptq",
        "quant_seconds": round(quant_s, 1),
        "n_calib": n_calib,
        "out_dir": str(out_dir),
        "generations": generations,
        "note": "N_CALIB=8 all-layer smoke. Not the Stage 2 N_CALIB=128 checkpoint.",
    }
    (out_dir / "gptq_tiny_smoke.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=("dispatch", "fp16", "gptq", "both"),
        default="dispatch",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda|cpu (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--full-weights",
        action="store_true",
        help="Load real google/gemma-2-9b (~18GB). Default fp16 uses a 1-layer geometry clone.",
    )
    p.add_argument(
        "--gptq-out",
        default=str(ROOT / "gemma2_gptq_tiny_layer0"),
    )
    args = p.parse_args()
    if args.device is None:
        import torch as _t

        args.device = "cuda" if _t.cuda.is_available() else "cpu"
    results = {}
    if args.stage in ("dispatch", "both", "fp16", "gptq"):
        results["dispatch"] = stage_dispatch()
    if args.stage in ("fp16", "both"):
        results["fp16"] = stage_fp16(device=args.device, full_weights=args.full_weights)
    if args.stage in ("gptq", "both"):
        gptq_out = Path(args.gptq_out)
        kw = Path("/kaggle/working")
        if gptq_out == ROOT / "gemma2_gptq_tiny_layer0" and kw.is_dir():
            gptq_out = kw / "gemma2_gptq_tiny_layer0"
        results["gptq"] = stage_gptq(out_dir=gptq_out, full_weights=args.full_weights)
    _log("RESULT " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "generations"} for k, v in results.items()}, indent=2, default=str))
    _log("ALL OK")


if __name__ == "__main__":
    main()
