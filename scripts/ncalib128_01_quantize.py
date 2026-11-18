"""Re-quantize Llama-3.1-8B base → GPTQ-INT4 with N_CALIB=128.

Byte-identical to juice_repro/_make_quantize_base_nb.py except:
  - N_CALIB = 128 (was 64 smoke)
  - OUT_DIR = /kaggle/working/llama31_8b_base_gptq_int4_ncalib128
    (does NOT overwrite llama31_8b_base_gptq_int4)

Recipe locked: gptqmodel 2.2, transformers 4.51.3, 4-bit, group_size=128,
sym=True, desc_act=True, C4 truncated to 512 tokens, BACKEND.TORCH on T4.
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
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
    print("transformers", transformers.__version__, "gptqmodel", gptqmodel.__version__)
    assert torch.cuda.is_available()
    assert transformers.__version__.startswith("4.51."), transformers.__version__

    _orig_llama_layer_fwd = LlamaDecoderLayer.forward

    def _llama_layer_fwd_tuple(self, *args, **kwargs):
        out = _orig_llama_layer_fwd(self, *args, **kwargs)
        if torch.is_tensor(out):
            return (out,)
        return out

    LlamaDecoderLayer.forward = _llama_layer_fwd_tuple
    print("LlamaDecoderLayer.forward patched to always return a tuple")

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

    FP16_MODEL_ID = "meta-llama/Llama-3.1-8B"
    # Versioned path: do NOT write to llama31_8b_base_gptq_int4 (N_CALIB=64).
    OUT_DIR = Path("/kaggle/working/llama31_8b_base_gptq_int4_ncalib128")

    N_CALIB = 128  # was 64 smoke; this is the only recipe change
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
        print("WARNING: prefer T4 x2 for 8B FP16 quant")

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
    print(
        "tokenized n_calib",
        len(calib_examples),
        "seq_len min/median/max",
        min(lens),
        sorted(lens)[len(lens) // 2],
        max(lens),
    )
    assert max(lens) <= MAX_SEQ_LEN
    elapsed("calibration data ready")

    gc.collect()
    torch.cuda.empty_cache()
    print("GPU before load: allocated %.2f GiB" % (torch.cuda.memory_allocated() / 1024**3))

    t0 = time.time()
    print("Loading FP16 for quantization (weights on CPU):", FP16_MODEL_ID)

    model = GPTQModel.load(
        FP16_MODEL_ID,
        quant_config,
        trust_remote_code=True,
    )

    print(
        "GPU after load: allocated %.2f GiB (expect <1 GiB; full model on CPU)"
        % (torch.cuda.memory_allocated() / 1024**3)
    )

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
        "note": "N_CALIB=128 re-quant; do not confuse with llama31_8b_base_gptq_int4 (N_CALIB=64)",
    }
    (OUT_DIR / "quant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Wrote quant_meta.json")
    elapsed("save done")

    try:
        from gptqmodel import BACKEND

        qmodel = GPTQModel.load(str(OUT_DIR), backend=BACKEND.TORCH)
    except Exception:
        qmodel = GPTQModel.load(str(OUT_DIR))

    prompt = "The capital of France is"
    with torch.inference_mode():
        result = qmodel.generate(prompt, max_new_tokens=16)
    text = result[0] if isinstance(result, (list, tuple)) else str(result)
    print("OUT:", text)
    assert len(text.strip()) > len(prompt), "Smoke failed"
    print("SMOKE OK")
    del qmodel
    gc.collect()
    torch.cuda.empty_cache()
    elapsed("smoke test done")
    print("Done. Download:", OUT_DIR)
    print("Do not overwrite llama31_8b_base_gptq_int4 (N_CALIB=64).")


if __name__ == "__main__":
    main()
