"""Quantize google/gemma-2-9b base → GPTQ-INT4 with N_CALIB=128.

Locked recipe (same as Llama Stage 2): gptqmodel 2.2, transformers 4.51.3,
4-bit, group_size=128, sym=True, desc_act=True, C4 truncated to 512 tokens,
BACKEND.TORCH on T4.

Quantize only. No FP16 JuICE wrap. Do not use gemma2_gptq_ncalib8.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE_T0 = time.time()


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def to_text(result, tokenizer) -> str:
    import torch

    if isinstance(result, (list, tuple)):
        result = result[0]
    if torch.is_tensor(result):
        ids = result.detach().cpu()
        if ids.ndim > 1:
            ids = ids[0]
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    if isinstance(result, str):
        return result
    return str(result)


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
            "datasets>=2.14.0",
            "sentencepiece",
            "protobuf",
            "tqdm",
        ]
    )
    elapsed("pip install")

    def _patch_no_init_weights() -> None:
        import transformers.modeling_utils as mu

        if hasattr(mu, "no_init_weights"):
            return
        try:
            from transformers.initialization import no_init_weights as _niw

            mu.no_init_weights = _niw
            print("patched no_init_weights from transformers.initialization")
            return
        except Exception:
            pass
        from contextlib import contextmanager

        @contextmanager
        def _noop(_enable=True):
            yield

        mu.no_init_weights = _noop
        print("patched no_init_weights with noop shim")

    _patch_no_init_weights()

    import torch
    import transformers
    import gptqmodel

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
    print("transformers", transformers.__version__, "gptqmodel", gptqmodel.__version__)
    assert torch.cuda.is_available()
    assert transformers.__version__.startswith("4.51."), transformers.__version__

    try:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

        _orig = Gemma2DecoderLayer.forward

        def _gemma2_layer_fwd_tuple(self, *args, **kwargs):
            out = _orig(self, *args, **kwargs)
            if torch.is_tensor(out):
                return (out,)
            return out

        Gemma2DecoderLayer.forward = _gemma2_layer_fwd_tuple
        print("Gemma2DecoderLayer.forward patched to always return a tuple")
    except ImportError:
        print("WARNING: no Gemma2DecoderLayer")

    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient

    login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
    print("HF login ok")
    elapsed("HF login")

    from gptqmodel import GPTQModel

    try:
        from gptqmodel import QuantizeConfig
    except ImportError:
        from gptqmodel import GPTQConfig as QuantizeConfig

    FP16_MODEL_ID = "google/gemma-2-9b"
    OUT_DIR = Path("/kaggle/working/gemma2_9b_base_gptq_int4_ncalib128")

    N_CALIB = 128
    MAX_SEQ_LEN = 512
    QUANT_BATCH_SIZE = 1
    QUANT_DEVICE = "cuda:0"

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        damp_percent=0.01,
        parallel_packing=False,
        device=QUANT_DEVICE,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("OUT_DIR", OUT_DIR)
    print("quant_config", quant_config)
    print("N_CALIB", N_CALIB)
    print("Do not use gemma2_gptq_ncalib8 / gemma2_gptq_tiny_layer0")

    existing_weights = list(OUT_DIR.glob("*.safetensors")) + list(OUT_DIR.glob("*.bin"))
    if (OUT_DIR / "config.json").is_file() and existing_weights:
        print("Checkpoint already present at", OUT_DIR, "— skipping quantize.")
        elapsed("quantize skipped (exists)")
        return

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU[{i}] {p.name} {p.total_memory / 1024**3:.1f} GiB")

    for label, path in [("working", "/kaggle/working"), ("tmp", "/tmp")]:
        total, used, free = shutil.disk_usage(path)
        print(f"disk {label}: free {free / 1024**3:.1f} GiB / {total / 1024**3:.1f} GiB")

    if torch.cuda.device_count() < 2:
        print("WARNING: prefer T4 x2 for 9B FP16 quant")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading C4 calibration shard...")
    ds = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
        trust_remote_code=True,
    )
    calib_texts = ds.select(range(min(N_CALIB, len(ds))))["text"]
    print("raw n_calib", len(calib_texts), "example_chars", len(calib_texts[0]))

    tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL_ID, trust_remote_code=True)
    calib_examples = []
    raw_chars = []
    for text in calib_texts:
        raw_chars.append(len(text))
        tok = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        calib_examples.append(
            {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
            }
        )

    lens = [ex["input_ids"].shape[1] for ex in calib_examples]
    calib_stats = {
        "n_calib": len(calib_examples),
        "max_seq_len": MAX_SEQ_LEN,
        "seq_len_min": int(min(lens)),
        "seq_len_median": int(sorted(lens)[len(lens) // 2]),
        "seq_len_max": int(max(lens)),
        "seq_len_mean": float(sum(lens) / len(lens)),
        "raw_chars_min": int(min(raw_chars)),
        "raw_chars_median": int(sorted(raw_chars)[len(raw_chars) // 2]),
        "raw_chars_max": int(max(raw_chars)),
        "calib_dataset": "allenai/c4 en/c4-train.00001-of-01024.json.gz",
        "calib_shard_slice": f"range({min(N_CALIB, len(ds))})",
    }
    print("=== calibration stats ===")
    print(json.dumps(calib_stats, indent=2))
    assert max(lens) <= MAX_SEQ_LEN

    del ds, calib_texts
    gc.collect()
    elapsed("calibration data ready")

    def _ram(tag: str) -> None:
        try:
            txt = Path("/proc/meminfo").read_text(encoding="utf-8")
            kv = {}
            for line in txt.splitlines():
                k, v = line.split(":", 1)
                kv[k] = v.strip()
            print(
                f"RAM {tag}: MemAvailable {kv.get('MemAvailable')} | "
                f"MemFree {kv.get('MemFree')} | MemTotal {kv.get('MemTotal')}"
            )
        except OSError:
            print("RAM", tag, "(no /proc/meminfo)")

    gc.collect()
    torch.cuda.empty_cache()
    _ram("before load")
    print("GPU before load: allocated %.2f GiB" % (torch.cuda.memory_allocated() / 1024**3))

    t0 = time.time()
    # Do NOT pass torch_dtype=float16. Gemma-2 checkpoints are bf16; a cast
    # can peak at 2× ~18GB and SIGKILL Kaggle RAM (Version 2, ~34 min).
    print("Loading for quantization (native dtype, CPU weights):", FP16_MODEL_ID)
    try:
        model = GPTQModel.load(
            FP16_MODEL_ID,
            quant_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = GPTQModel.load(FP16_MODEL_ID, quant_config, trust_remote_code=True)

    print(
        "GPU after load: allocated %.2f GiB (expect <1 GiB; full model on CPU)"
        % (torch.cuda.memory_allocated() / 1024**3)
    )
    _ram("after load")

    print(
        "Starting GPTQ quantize batch_size=%d max_seq=%d gpu_cache=False..."
        % (QUANT_BATCH_SIZE, MAX_SEQ_LEN)
    )
    model.quantize(
        calib_examples,
        batch_size=QUANT_BATCH_SIZE,
        calibration_enable_gpu_cache=False,
        auto_gc=True,
        buffered_fwd=False,
    )
    quant_seconds = time.time() - t0
    print(f"Quantize done in {quant_seconds / 60:.1f} min ({quant_seconds:.1f} s)")
    elapsed("quantize done")

    model.save(str(OUT_DIR))
    print("Saved to", OUT_DIR)
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".safetensors", ".bin"):
            print(f"  {p.relative_to(OUT_DIR)}  {p.stat().st_size / 1024**2:.1f} MiB")

    meta = {
        "fp16_source": FP16_MODEL_ID,
        "quant_method": "gptq",
        "tool": f"gptqmodel=={gptqmodel.__version__}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bits": 4,
        "group_size": 128,
        "desc_act": True,
        "sym": True,
        "damp_percent": 0.01,
        "n_calib": len(calib_examples),
        "max_seq_len": MAX_SEQ_LEN,
        "calib_dataset": "allenai/c4 en/c4-train.00001-of-01024.json.gz",
        "quant_batch_size": QUANT_BATCH_SIZE,
        "quant_device": QUANT_DEVICE,
        "calibration_enable_gpu_cache": False,
        "output_dir": str(OUT_DIR),
        "quant_seconds": quant_seconds,
        "quant_minutes": quant_seconds / 60.0,
        "calib_stats": calib_stats,
        "note": "Canonical Gemma-2-9B INT4 for Stage 2. Not the N_CALIB=8 smoke.",
    }
    (OUT_DIR / "quant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Wrote quant_meta.json")
    elapsed("save done")
    _ram("after save+del")
    print("=" * 60)
    print("CHECKPOINT ON DISK. If the TORCH smoke SIGKILLs, Quick Save this draft.")
    print("folder:", OUT_DIR)
    print("=" * 60)

    from gptqmodel import BACKEND

    qmodel = GPTQModel.load(str(OUT_DIR), backend=BACKEND.TORCH)
    print("loaded BACKEND.TORCH")
    tok = getattr(qmodel, "tokenizer", None) or tokenizer

    prompts = [
        "The capital of France is",
        "Question: What is 2 + 2?\nAnswer:",
        "Once upon a time",
    ]
    garbage = []
    generations = []
    for p in prompts:
        with torch.inference_mode():
            result = qmodel.generate(p, max_new_tokens=16)
        text = to_text(result, tok)
        generations.append({"prompt": p, "out": text})
        print("GEN", repr(p), "->", repr(text))
        body = text[len(p) :] if text.startswith(p) else text
        compact = "".join(body.split())
        if len(compact) < 3 or compact == compact[0] * len(compact):
            garbage.append(p)
    if garbage:
        raise AssertionError(f"garbage generations: {garbage}")
    print("SMOKE OK (decoded text, BACKEND.TORCH)")
    (OUT_DIR / "decode_smoke.json").write_text(
        json.dumps({"ok": True, "generations": generations}, indent=2), encoding="utf-8"
    )
    del qmodel
    gc.collect()
    torch.cuda.empty_cache()
    elapsed("smoke test done")
    print("Done. Checkpoint:", OUT_DIR)


if __name__ == "__main__":
    main()
