"""Greedy decoding helpers (batch size 1, memory-conscious)."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from common.models import input_device
from common.prompts import ConflictItem, Condition, apply_chat_template


@torch.inference_mode()
def greedy_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 32,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    # model.device is wrong / ambiguous under device_map="auto" (sharded 8B FP16).
    dev = input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    # Diagnostic: inspect if any parameters or buffers are left on the 'meta' device
    meta_params = [(name, p.shape, p.dtype) for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [(name, b.shape, b.dtype) for name, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params or meta_buffers:
        print(f"\n{'!'*80}")
        print(f"[DIAGNOSTIC] FOUND UNMATERIALIZED META TENSORS BEFORE generate():")
        for name, shape, dtype in meta_params:
            print(f"  [PARAM ON META] {name}: shape={shape}, dtype={dtype}")
        for name, shape, dtype in meta_buffers:
            print(f"  [BUFFER ON META] {name}: shape={shape}, dtype={dtype}")
        print(f"[DIAGNOSTIC] Total meta params: {len(meta_params)}, meta buffers: {len(meta_buffers)}")
        print(f"[DIAGNOSTIC] input_device: {dev}, model.device: {getattr(model, 'device', 'N/A')}")
        print(f"{'!'*80}\n", flush=True)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_for_item(
    model,
    tokenizer,
    item: ConflictItem,
    condition: Condition,
    *,
    max_new_tokens: int = 32,
) -> str:
    prompt = apply_chat_template(tokenizer, item, condition)
    return greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
