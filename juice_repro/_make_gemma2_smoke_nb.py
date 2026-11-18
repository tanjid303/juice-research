"""Generate juice_repro/kaggle_gemma2_juice_smoke.ipynb

Gemma-2-9B JuICE port smokes. Do not start N_CALIB=128 until GPTQ gen is clean.
"""
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
    """# Gemma-2-9B JuICE port smoke (FP16 wrap + tiny GPTQ)

Prove the Gemma-2 port on Kaggle **before** any `N_CALIB=128` quantize.

1. Full `google/gemma-2-9b` **base** eager FP16 wrap + one dual-run
2. Tiny GPTQ: 1 layer / 8 calib samples, `BACKEND.TORCH`, greedy decode **not garbage**

**Do not** use the tiny GPTQ folder for Stage 2. If decode is garbage or wrap cannot share `o_proj` with `TorchQuantLinear`, pivot to Qwen2.5-7B.

**~20–40 min** on GPU T4 ×2 (download + FP16 forward + 1-layer GPTQ).

## Before Run All

1. **GPU T4 ×2**, Internet **ON**, secret `HF_TOKEN`
2. Accept [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) license (gated)
3. Attach **`kaggle_juice_gemma2_smoke.zip`** as a dataset
4. Save & Run All
5. Pass = config `256 / 16 / 8 / 50`, 42 wrapped layers, `head_output` `[B, S, 3584]`, GPTQ gens are real words
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

# Same pin as Llama GPTQ: 4.56+/5.x decoder layers return a Tensor; gptqmodel 2.2
# then indexes [0] and drops batch. Kaggle ships transformers 5.x.
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
assert torch.cuda.is_available(), "Need GPU T4 x2"
assert transformers.__version__.startswith("4.51."), transformers.__version__
if torch.cuda.device_count() < 2:
    print("WARNING: prefer T4 x2 — Gemma-2-9B FP16 is ~18GB")

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
    print("no Gemma2DecoderLayer (unexpected on transformers 4.51)")

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
print("HF login ok")
"""
)

md("## 2 — Locate zip + GPU / disk")
code(
    r"""def find_one(name: str) -> Path:
    hits = list(Path("/kaggle/working").rglob(name))
    if Path("/kaggle/input").exists():
        hits += list(Path("/kaggle/input").rglob(name))
    hits = [h for h in hits if h.is_file()]
    assert hits, f"Missing {name} — attach kaggle_juice_gemma2_smoke.zip as a dataset"
    hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))
    print(name, "->", hits[0])
    return hits[0]

SMOKE = find_one("gemma2_juice_smoke.py")
ROOT = SMOKE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
print("ROOT", ROOT)
print("data", (ROOT / "data" / "conflictqa_pilot250_popular.jsonl").is_file())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU[{i}] {p.name} {p.total_memory/1024**3:.1f} GiB")

for label, path in [("working", "/kaggle/working"), ("tmp", "/tmp")]:
    total, used, free = shutil.disk_usage(path)
    print(f"disk {label}: free {free/1024**3:.1f} GiB / {total/1024**3:.1f} GiB")
"""
)

md("## 3 — FP16 full 9B wrap + dual-run")
code(
    r"""print("=== STAGE fp16 --full-weights ===")
t0 = time.time()
subprocess.check_call(
    [
        sys.executable,
        "-u",
        str(SMOKE),
        "--stage",
        "fp16",
        "--full-weights",
    ],
    cwd=str(ROOT),
    env={**os.environ, "PYTHONPATH": str(ROOT)},
)
print(f"=== STAGE fp16 done in {(time.time()-t0)/60:.1f} min ===")
"""
)

md("## 4 — GPTQ N_CALIB=8 (all layers, in-process). Not N_CALIB=128.")
code(
    r"""# In-process on purpose. A subprocess doubles CUDA + 18GB Gemma and SIGKILLs Kaggle RAM.
# Skip this cell in the same session as the FP16 smoke. Fresh session: cells 1–2, skip 3, then this.

import gc
gc.collect()
torch.cuda.empty_cache()

GPTQ_OUT = Path("/kaggle/working/gemma2_gptq_ncalib8")
if GPTQ_OUT.exists():
    shutil.rmtree(GPTQ_OUT)
GPTQ_OUT.mkdir(parents=True)

from gptqmodel import GPTQModel
try:
    from gptqmodel import QuantizeConfig
except ImportError:
    from gptqmodel import GPTQConfig as QuantizeConfig
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-2-9b"
calib_texts = [
    "The capital of France is Paris, a city on the Seine.",
    "Gemma is a family of open language models released by Google.",
    "Question: Who wrote Pride and Prejudice?\nAnswer: Jane Austen.",
    "Sliding window attention alternates with global attention layers.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Python is a programming language used in scientific computing.",
    "A transformer decoder layer contains self-attention and an MLP.",
    "Calibration text for a Gemma-2 GPTQ smoke test.",
]
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
calib = []
for text in calib_texts:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    calib.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

quant_config = QuantizeConfig(
    bits=4, group_size=128, desc_act=True, sym=True,
    damp_percent=0.01, parallel_packing=False, device="cuda:0",
)
print("Loading Gemma-2-9B on CPU for GPTQ (expect ~18GB RAM). Do not have FP16 loaded.")
t0 = time.time()
try:
    model = GPTQModel.load(MODEL_ID, quant_config, trust_remote_code=True, torch_dtype=torch.float16)
except TypeError:
    model = GPTQModel.load(MODEL_ID, quant_config, trust_remote_code=True)
print("load ok, GPU allocated %.2f GiB" % (torch.cuda.memory_allocated() / 1024**3))
model.quantize(calib, batch_size=1, calibration_enable_gpu_cache=False, auto_gc=True, buffered_fwd=False)
print(f"quantize done in {(time.time()-t0)/60:.1f} min")
model.save(str(GPTQ_OUT))
del model
gc.collect()
torch.cuda.empty_cache()

try:
    from gptqmodel import BACKEND
    qmodel = GPTQModel.load(str(GPTQ_OUT), backend=BACKEND.TORCH)
    print("loaded BACKEND.TORCH")
except Exception as exc:
    print("BACKEND.TORCH failed", exc)
    qmodel = GPTQModel.load(str(GPTQ_OUT))

prompts = [
    "The capital of France is",
    "Question: What is 2 + 2?\nAnswer:",
    "Once upon a time",
]
generations, garbage = [], []
for p in prompts:
    with torch.inference_mode():
        result = qmodel.generate(p, max_new_tokens=16)
    text = result[0] if isinstance(result, (list, tuple)) else str(result)
    generations.append({"prompt": p, "out": text})
    print("GEN", repr(p), "->", repr(text))
    body = text[len(p):] if text.startswith(p) else text
    compact = "".join(body.split())
    if len(compact) < 3 or compact == compact[0] * len(compact):
        garbage.append(p)
del qmodel
gc.collect()
torch.cuda.empty_cache()
assert not garbage, f"garbage generations: {garbage}"
meta = {
    "ok": True, "n_calib": 8, "out_dir": str(GPTQ_OUT),
    "generations": generations,
    "note": "N_CALIB=8 all-layer smoke. Not the Stage 2 N_CALIB=128 checkpoint.",
}
(GPTQ_OUT / "gptq_tiny_smoke.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("GPTQ smoke: generations look non-garbage")
"""
)

md("## 5 — Confirm, then stop. Do not N_CALIB=128 yet.")
code(
    r"""meta = GPTQ_OUT / "gptq_tiny_smoke.json"
print("tiny gptq dir", GPTQ_OUT, "exists" if GPTQ_OUT.is_dir() else "MISSING")
print("gptq_tiny_smoke.json", meta.is_file())
if meta.is_file():
    print(meta.read_text(encoding="utf-8")[:4000])

print()
print("=" * 60)
print("PASS CHECKLIST")
print("  config: head_dim=256, heads=16, kv=8, softcap=50, hidden=3584, layers=42")
print("  wrap: GetHeadsGemma2Attention, 42 layers, head_output [B, S, 3584]")
print("  dual_run_logits: printed ALL OK after fp16")
print("  GPTQ gens: real words, not aaaa / empty")
print()
print("If GPTQ gens are garbage or wrap/o_proj fails: pivot to Qwen2.5-7B.")
print("If both passed: next is N_CALIB=128 with the Llama recipe (do not use this tiny ckpt).")
print("=" * 60)
print("SAVE VERSION NOW (optional — only the json/log matters; discard tiny GPTQ weights)")
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

out = Path(__file__).resolve().parent / "kaggle_gemma2_juice_smoke.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("Wrote", out)
