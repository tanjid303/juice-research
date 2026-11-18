#!/usr/bin/env python3
"""Inspect GPTQ checkpoint load quality (quant config, tensors, generations, PPL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.models import GPTQ_MODEL_ID, load_model_and_tokenizer, unload_model, vram_mib

PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The chemical symbol for gold is",
    "William Shakespeare wrote",
    "The largest planet in the solar system is",
    "Photosynthesis occurs in the",
    "The speed of light is approximately",
    "DNA stands for",
]

PPL_TEXT = (
    "The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome "
    "that covers most of the Amazon basin of South America. This basin encompasses 7 million "
    "square kilometers, of which 5.5 million square kilometers are covered by the rainforest. "
    "This region includes territory belonging to nine nations and 3.4 million square kilometers "
    "of relatively intact forest are mostly contained within Brazil. The Amazon represents over "
    "half of the planet's remaining rainforests and comprises the largest and most biodiverse "
    "tract of tropical rainforest in the world, with an estimated 390 billion individual trees "
    "divided into 16,000 species. It is home to many species of plants and animals, and plays "
    "a critical role in regulating Earth's climate through carbon storage and the water cycle."
)


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and isinstance(getattr(obj, k), (int, float, str, bool, type(None)))}


def inspect_layer(model) -> dict:
    layer = model.model.layers[0].self_attn.q_proj
    info = {"module_type": type(layer).__name__, "attrs": []}
    for name in ("qweight", "qzeros", "scales", "g_idx", "bias", "weight"):
        if hasattr(layer, name):
            t = getattr(layer, name)
            if t is None:
                info["attrs"].append({"name": name, "present": True, "is_none": True})
            elif torch.is_tensor(t):
                unique_g = None
                sequential = None
                if name == "g_idx" and t.numel() > 0:
                    g = t.detach().cpu().long().flatten()
                    unique_g = int(g.unique().numel())
                    # Sequential grouping: floor(i / group_size) with group_size inferred
                    # from first change. Common default when desc_act=False.
                    diffs = (g[1:] != g[:-1]).nonzero(as_tuple=False)
                    sequential = True
                    if diffs.numel() == 0:
                        sequential = g[0].item() == 0
                    else:
                        # All runs of equal length except possibly last
                        idx = [0] + [int(x) + 1 for x in diffs.flatten().tolist()] + [g.numel()]
                        run_lens = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
                        sequential = all(run_lens[i] == run_lens[0] for i in range(len(run_lens) - 1))
                        if sequential:
                            expected = torch.arange(g.numel(), device=g.device) // run_lens[0]
                            sequential = bool(torch.equal(g, expected.to(g.dtype)))
                info["attrs"].append(
                    {
                        "name": name,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "numel": int(t.numel()),
                        "unique_g_idx": unique_g,
                        "g_idx_sequential": sequential,
                    }
                )
            else:
                info["attrs"].append({"name": name, "type": type(t).__name__})
    return info


@torch.inference_mode()
def generate_samples(model, tokenizer, max_new_tokens: int = 32) -> list[dict]:
    rows = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rows.append({"prompt": prompt, "output": gen.strip()})
    return rows


@torch.inference_mode()
def perplexity(model, tokenizer, text: str) -> dict:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    n_tokens = int(input_ids.shape[1])
    outputs = model(input_ids, labels=input_ids)
    loss = float(outputs.loss.item())
    ppl = float(torch.exp(outputs.loss).item())
    return {"n_tokens": n_tokens, "nll": loss, "ppl": ppl}


def main() -> None:
    print("=== Load INT4 GPTQ ===")
    model, tokenizer = load_model_and_tokenizer("int4", quant_backend="gptq")
    qcfg = _as_dict(getattr(model.config, "quantization_config", None))
    print("quantization_config:")
    print(json.dumps(qcfg, indent=2, default=str))
    layer_info = inspect_layer(model)
    print("layer0.q_proj:")
    print(json.dumps(layer_info, indent=2))
    gens = generate_samples(model, tokenizer)
    print("generations:")
    for g in gens:
        print(f"  Q: {g['prompt']}")
        print(f"  A: {g['output'][:300]!r}")
    int4_ppl = perplexity(model, tokenizer, PPL_TEXT)
    print("INT4 ppl:", int4_ppl, "peak_vram", vram_mib())
    unload_model(model)

    print("=== Load FP16 ===")
    model, tokenizer = load_model_and_tokenizer("fp16")
    fp16_ppl = perplexity(model, tokenizer, PPL_TEXT)
    print("FP16 ppl:", fp16_ppl, "peak_vram", vram_mib())
    unload_model(model)

    rel = (int4_ppl["ppl"] - fp16_ppl["ppl"]) / fp16_ppl["ppl"] if fp16_ppl["ppl"] else float("nan")
    report = {
        "model_id": GPTQ_MODEL_ID,
        "quantization_config": qcfg,
        "layer0_q_proj": layer_info,
        "generations": gens,
        "fp16_ppl": fp16_ppl,
        "int4_ppl": int4_ppl,
        "relative_ppl_increase": rel,
    }
    out = ROOT / "mini_pilot" / "outputs" / "gptq_quality_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"relative PPL increase: {rel:.3%}")


if __name__ == "__main__":
    main()
