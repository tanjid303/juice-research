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
