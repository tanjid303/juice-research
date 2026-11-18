"""Generate juice_repro/kaggle_quantize_llama31_8b_base_gptq.ipynb"""
import json
from pathlib import Path

cells = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.split("\n")]})


def code(s: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [line + "\n" for line in s.split("\n")],
        }
    )


md(
    """# GPTQ-INT4 — Llama-3.1-8B **base** (self-quantize on Kaggle)

Quantize `meta-llama/Llama-3.1-8B` → GPTQ INT4 for Stage 2 INT4 transfer.

**Kaggle Python 3.12 note:** Uses **`gptqmodel==2.2.0`** with **`transformers==4.51.3`**. Newer 4.56+/5.x Llama layers return a bare tensor; gptqmodel then indexes `[0]` and drops the batch dim (RoPE crash on layer 1).

**Recipe:** 4-bit, `group_size=128`, `sym=True`, `desc_act=True` (hugging-quants-compatible).

## Before Run All

1. **GPU T4 ×2**, Internet ON, `HF_TOKEN` secret
2. Accept [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) license
3. **Save Version** → download `/kaggle/working/llama31_8b_base_gptq_int4/`

**VRAM:** T4 ×2, weights load on **CPU**; one layer at a time on GPU. C4 texts are truncated to **512 tokens** (full docs OOM in attention). Start with `N_CALIB=64` smoke, then `128` / `512` for final.

**Plan B if quant still fails:** use community [ModelCloud/Meta-Llama-3.1-8B-gptq-4bit](https://huggingface.co/ModelCloud/Meta-Llama-3.1-8B-gptq-4bit) and skip self-quant.
"""
)

md("## 1 — Setup (gptqmodel 2.2.0 + transformers 4.51.3)")
code(
    r"""import os, sys, subprocess, json, shutil, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Pin 4.51.3: LlamaDecoderLayer must return a tuple. 4.56+ returns a Tensor;
# gptqmodel does layer_output[0] and drops batch → RoPE 32 vs 128 on layer 1.
# Also needed because Kaggle ships transformers 5.x (breaks gptqmodel 2.2 import).
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

# Shim if an old gptqmodel build still imports modeling_utils.no_init_weights
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

# gptqmodel 2.2 takes layer_output[0]. A Tensor would slice batch; wrap if needed.
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
"""
)

md("## 2 — Config")
code(
    r"""from gptqmodel import GPTQModel

try:
    from gptqmodel import QuantizeConfig
except ImportError:
    from gptqmodel import GPTQConfig as QuantizeConfig

FP16_MODEL_ID = "meta-llama/Llama-3.1-8B"
OUT_DIR = Path("/kaggle/working/llama31_8b_base_gptq_int4")

N_CALIB = 64          # smoke: 64 → 128 → 512 for final checkpoint
MAX_SEQ_LEN = 512     # truncate C4 (full docs = 10k+ tokens → OOM in attention)
QUANT_BATCH_SIZE = 1
QUANT_DEVICE = "cuda:0"  # layer-wise quant GPU; weights stay on CPU

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    damp_percent=0.01,
    parallel_packing=False,  # lower peak VRAM during pack
    device=QUANT_DEVICE,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR", OUT_DIR)
print("quant_config", quant_config)
print("N_CALIB", N_CALIB)
"""
)

md("## 3 — GPU / disk sanity")
code(
    r"""for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU[{i}] {p.name} {p.total_memory/1024**3:.1f} GiB")

for label, path in [("working", "/kaggle/working"), ("tmp", "/tmp")]:
    total, used, free = shutil.disk_usage(path)
    print(f"disk {label}: free {free/1024**3:.1f} GiB / {total/1024**3:.1f} GiB")

if torch.cuda.device_count() < 2:
    print("WARNING: prefer T4 x2 for 8B FP16 quant")
"""
)

md("## 4 — Calibration data (C4, truncated)")
code(
    r"""from datasets import load_dataset
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

# Must truncate: tokenizer(text) without max_length keeps full C4 docs → OOM in SDPA
tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL_ID, trust_remote_code=True)
calib_examples = []
for text in calib_texts:
    tok = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    calib_examples.append({
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
    })

lens = [ex["input_ids"].shape[1] for ex in calib_examples]
print(
    "tokenized n_calib", len(calib_examples),
    "seq_len min/median/max", min(lens), sorted(lens)[len(lens) // 2], max(lens),
)
assert max(lens) <= MAX_SEQ_LEN
"""
)

md("## 5 — Quantize (layer-wise, CPU weights)")
code(
    r"""import gc

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

print("GPU after load: allocated %.2f GiB (expect <1 GiB; full model on CPU)" % (torch.cuda.memory_allocated() / 1024**3))

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
print(f"Quantize done in {(time.time()-t0)/60:.1f} min")
"""
)

md("## 6 — Save")
code(
    r"""model.save(str(OUT_DIR))
print("Saved to", OUT_DIR)
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file() and p.suffix in (".json", ".safetensors", ".bin"):
        print(f"  {p.relative_to(OUT_DIR)}  {p.stat().st_size/1024**2:.1f} MiB")

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
}
(OUT_DIR / "quant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

del model
import gc
gc.collect()
torch.cuda.empty_cache()
print("Wrote quant_meta.json")
"""
)

md("## 7 — Smoke test")
code(
    r"""try:
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
"""
)

md("## 8 — Next steps")
code(
    r"""print("Done. Download:", OUT_DIR)
print("Re-upload as Kaggle dataset for INT4 JuICE transfer.")
print("Set GPTQ_8B_BASE_MODEL_ID to that path in common/models.py")
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "gpu",
    },
    "cells": cells,
}
out = Path(__file__).resolve().parent / "kaggle_quantize_llama31_8b_base_gptq.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
