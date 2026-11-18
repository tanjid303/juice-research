"""Model loading with explicit VRAM cleanup between precisions."""

from __future__ import annotations

import gc
import inspect
import sys
import time
from typing import Literal, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Precision = Literal["fp16", "int4"]
QuantBackend = Literal["gptq", "bnb_nf4"]
DeviceMapArg = Union[str, dict, None]

FP16_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# Community GPTQ checkpoint; swap if you quantize your own copy.
GPTQ_MODEL_ID = "fbaldassarri/meta-llama_Llama-3.2-1B-Instruct-auto_gptq-int4-gs128-sym"

# Llama-3.1-8B (Kaggle T4 x2 / A4500). T4 has no useful native bf16 — keep float16.
FP16_8B_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
FP16_8B_BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"
# Vanilla AutoGPTQ INT4, group_size=128, desc_act=True (not AutoRound-in-GPTQ-clothing).
GPTQ_8B_MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
# Self-quantized Llama-3.1-8B base GPTQ-INT4 (Kaggle quant notebook output).
# Same session: /kaggle/working/llama31_8b_base_gptq_int4
# Next notebook: attach that output (Add Data → Your Notebooks) or re-upload as a dataset.
GPTQ_8B_BASE_MODEL_ID = "/kaggle/working/llama31_8b_base_gptq_int4"
GPTQ_8B_BASE_MODEL_ID_FALLBACK = "ModelCloud/Meta-Llama-3.1-8B-gptq-4bit"
# Canonical Gemma-2-9B base GPTQ-INT4 (N_CALIB=128). Never the N_CALIB=8 smoke.
GPTQ_GEMMA2_9B_BASE_N128 = "/kaggle/working/gemma2_9b_base_gptq_int4_ncalib128"


def input_device(model) -> torch.device:
    """First real device for inputs when the model is sharded via device_map='auto'."""
    dmap = getattr(model, "hf_device_map", None)
    if isinstance(dmap, dict) and dmap:
        for val in dmap.values():
            if val in ("cpu", "disk", None):
                continue
            if isinstance(val, int):
                return torch.device(f"cuda:{val}")
            return torch.device(val)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_max_memory(*, gpu_fraction: float = 0.90, forbid_cpu_offload: bool = True) -> dict:
    """
    Per-GPU cap for accelerate device_map='auto'.

    Leave ~10% headroom for CUDA context + activations. Setting cpu=0 makes a
    too-big load fail loudly instead of crawling on CPU/disk (the 4GB trap).
    """
    max_memory: dict = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = int(total * gpu_fraction)
    if forbid_cpu_offload:
        max_memory["cpu"] = 0
    return max_memory


def resolve_device_map(
    device_map: DeviceMapArg,
    *,
    precision: Precision,
    quant_backend: QuantBackend = "gptq",
) -> str | dict:
    """
    Default remains single-GPU {"": 0} (1B / 4GB safety).

    8B FP16 on Kaggle T4 x2: pass device_map='auto' (weights ~16.1 GiB; one 16GB
    T4 cannot hold them plus CUDA/activations).

    8B GPTQ-INT4 (~5–7 GiB): keep {"": 0}. AutoGPTQ kernels are reliable when
    every QuantLinear lives on one GPU. device_map='auto' *is* supported (HF
    shards at layer boundaries) and will usually park the whole INT4 model on
    GPU 0 anyway because it fits — but pinning GPU 0 avoids a leftover split
    after an FP16 two-GPU session.
    """
    if device_map is not None:
        return device_map
    if not torch.cuda.is_available():
        return "cpu"
    return {"": 0}


_CUDA_LEFTOVER_NAMES = (
    "model",
    "tokenizer",
    "processor",
    "qmodel",
    "wrapped",
    "ident",
    "ident_ctx",
    "ident_sup",
    "ident_rand",
)


def force_release_cuda(tag: str = "release") -> None:
    """Drop leftover model/tokenizer names, then empty + synchronize CUDA.

    Notebook cells that `exec` a runner in-process can keep INT4 weights alive
    through lingering Python references. Call this before every precision load.
    Prefer a separate Python process for H− so the CUDA context itself dies.
    """
    namespaces = []
    if "__main__" in sys.modules:
        namespaces.append(getattr(sys.modules["__main__"], "__dict__", {}))
    fr = inspect.currentframe()
    try:
        caller = fr.f_back if fr is not None else None
        while caller is not None:
            namespaces.append(caller.f_globals)
            namespaces.append(caller.f_locals)
            caller = caller.f_back
    finally:
        del fr
    for ns in namespaces:
        if not isinstance(ns, dict):
            continue
        for key in list(ns):
            if key in _CUDA_LEFTOVER_NAMES:
                ns.pop(key, None)
    gc.collect()
    if not torch.cuda.is_available():
        print(f"CUDA {tag}: unavailable")
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    time.sleep(0.5)
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(
            f"CUDA {tag}: gpu{i} allocated={alloc:.2f} GiB "
            f"reserved={reserved:.2f} GiB"
        )


def unload_model(model, tokenizer=None, *extra) -> None:
    """Drop model + tokenizer references and clear CUDA cache."""
    dmap = getattr(model, "hf_device_map", None)
    devices = {str(v) for v in dmap.values()} if isinstance(dmap, dict) else set()
    # 8B FP16 sharded across GPUs: .to("cpu") would copy ~16GB into RAM for no gain.
    skip_to_cpu = len(devices) > 1 or any(
        d not in ("0", "cuda:0") and "cpu" not in d for d in devices
    )
    try:
        if not skip_to_cpu and hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    for obj in extra:
        try:
            del obj
        except Exception:
            pass
    if tokenizer is not None:
        try:
            del tokenizer
        except Exception:
            pass
    del model
    force_release_cuda("after unload_model")


def _repair_gptq_g_idx_if_needed(model) -> None:
    """
    AutoRound/GPTQ exports with desc_act=False often omit g_idx. HF then leaves a
    zero buffer. cuda-old dequant uses group_size so zeros are harmless there, but
    kernels that index scales[g_idx] would be wrong. Reconstruct sequential groups.
    """
    qcfg = getattr(model.config, "quantization_config", None)
    desc_act = bool(getattr(qcfg, "desc_act", False)) if qcfg is not None else False
    if desc_act:
        return
    repaired = 0
    for module in model.modules():
        if not (hasattr(module, "g_idx") and hasattr(module, "group_size")):
            continue
        g = module.g_idx
        if g is None or not torch.is_tensor(g) or g.numel() == 0:
            continue
        if not bool(torch.all(g == 0)):
            continue
        gs = int(module.group_size)
        in_features = int(g.numel())
        module.g_idx.copy_(
            torch.arange(in_features, device=g.device, dtype=g.dtype) // gs
        )
        repaired += 1
    if repaired:
        print(
            f"Repaired sequential g_idx on {repaired} GPTQ layers "
            "(desc_act=False, export omitted g_idx)."
        )


def _device_from_map(device_map: DeviceMapArg) -> str:
    if device_map is None or device_map == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device_map, dict):
        if "" in device_map:
            val = device_map[""]
            if isinstance(val, int):
                return f"cuda:{val}"
            return str(val)
        for val in device_map.values():
            if val in ("cpu", "disk", None):
                continue
            if isinstance(val, int):
                return f"cuda:{val}"
            return str(val)
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sanitize_gptq_configs_for_gptqmodel(local_dir) -> None:
    """
    Older AutoGPTQ exports set `is_marlin_format`. GPTQModel >=7 rejects that key.
    Drop it (and map True -> format=marlin when format is missing).
    """
    import json
    from pathlib import Path

    local = Path(local_dir)

    def scrub(obj: dict) -> bool:
        changed = False
        if "is_marlin_format" in obj:
            is_marlin = bool(obj.pop("is_marlin_format"))
            if is_marlin and "format" not in obj:
                obj["format"] = "marlin"
            changed = True
        nested = obj.get("quantization_config")
        if isinstance(nested, dict) and scrub(nested):
            changed = True
        return changed

    for name in ("quantize_config.json", "config.json"):
        path = local / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and scrub(data):
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Sanitized {name}: removed is_marlin_format for gptqmodel")


def _looks_like_gptq_dir(path) -> bool:
    from pathlib import Path

    p = Path(path)
    if not p.is_dir() or not (p / "config.json").is_file():
        return False
    names = {n.name.lower() for n in p.iterdir() if n.is_file()}
    has_weights = any(n.endswith(".safetensors") or n.endswith(".bin") for n in names)
    if not has_weights:
        return False
    if "quantize_config.json" in names or "quant_meta.json" in names:
        return True
    cfg = (p / "config.json").read_text(encoding="utf-8", errors="ignore").lower()
    return "gptq" in cfg or "quantization_config" in cfg


def list_kaggle_model_inputs(*, max_files: int = 80) -> list[str]:
    """Print a short inventory of /kaggle/input (and /kaggle/working model dirs)."""
    from pathlib import Path

    lines: list[str] = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            lines.append(f"{root}  MISSING")
            continue
        lines.append(f"{root}  exists")
        try:
            tops = sorted(root.iterdir())
        except OSError as e:
            lines.append(f"  (listdir failed: {e})")
            continue
        for top in tops[:40]:
            kind = "DIR " if top.is_dir() else "FILE"
            lines.append(f"  {kind} {top.name}")
        n = 0
        try:
            for p in root.rglob("config.json"):
                parent = p.parent
                weight = any(
                    x.suffix in {".safetensors", ".bin"} for x in parent.iterdir() if x.is_file()
                )
                lines.append(f"  config.json -> {parent}  weights={weight}")
                n += 1
                if n >= max_files:
                    break
        except OSError as e:
            lines.append(f"  (rglob failed: {e})")
    return lines


def resolve_gptq_8b_base_model_id(explicit: str | None = None) -> str:
    """
    Find the self-quantized Llama-3.1-8B base GPTQ checkpoint.

    Search order: explicit path → GPTQ_8B_BASE_MODEL_ID → Kaggle working/input
    dirs named llama31_8b_base_gptq_int4 → folders with config.json + GPTQ weights.
    """
    from pathlib import Path

    candidates: list[Path] = []
    for raw in (explicit, GPTQ_8B_BASE_MODEL_ID):
        if raw:
            candidates.append(Path(raw))
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if not root.exists():
            continue
        candidates.append(root / "llama31_8b_base_gptq_int4")
        try:
            for name in ("quantize_config.json", "quant_meta.json"):
                for cfg in root.rglob(name):
                    candidates.append(cfg.parent)
            for cfg in root.rglob("config.json"):
                candidates.append(cfg.parent)
            for d in root.rglob("llama31_8b_base_gptq_int4"):
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
            print("Resolved GPTQ_8B_BASE_MODEL_ID ->", path)
            return str(path)

    inventory = "\n".join(list_kaggle_model_inputs())
    raise FileNotFoundError(
        "Self-quantized base GPTQ not found.\n"
        "This notebook only has the code zip. You must ALSO attach the quantize "
        "notebook OUTPUT (Add Data → Notebook Output Files → that Save Version), "
        "or upload llama31_8b_base_gptq_int4 as a dataset.\n"
        "Then set INT4_MODEL_ID_OVERRIDE to the folder that contains config.json "
        "and *.safetensors.\n\n"
        f"Tried {list(seen)[:12]}\n\nInventory:\n{inventory}"
    )


def resolve_gemma2_ncalib128_checkpoint(explicit: str | None = None) -> str:
    """Only accept gemma2_9b_base_gptq_int4_ncalib128. Never N_CALIB=8 / 1-layer smoke."""
    from pathlib import Path

    named = "gemma2_9b_base_gptq_int4_ncalib128"
    forbidden = ("ncalib8", "tiny_layer", "gptq_tiny", "layer0")
    candidates: list[Path] = []
    env = __import__("os").environ.get("GEMMA2_INT4_CKPT")
    for raw in (explicit, env, GPTQ_GEMMA2_9B_BASE_N128):
        if raw:
            candidates.append(Path(raw))
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
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
        low = resolved.replace("\\", "/").lower()
        if any(tok in low for tok in forbidden) and "ncalib128" not in low:
            print("skip non-canonical Gemma GPTQ", path)
            continue
        if _looks_like_gptq_dir(path) and named in low:
            print("Resolved Gemma N_CALIB=128 GPTQ ->", path)
            return str(path)
    inventory = "\n".join(list_kaggle_model_inputs())
    raise FileNotFoundError(
        "Gemma N_CALIB=128 checkpoint not found. Expected a folder named "
        f"{named} with config.json + *.safetensors.\n"
        "Attach the gemma-quantize notebook output (Add Input → Notebook Output Files).\n"
        "Do not use gemma2_gptq_ncalib8.\n"
        f"Tried {list(seen)[:12]}\n\nInventory:\n{inventory}"
    )


def _gptq_config_model_type(local_dir: str) -> str | None:
    import json
    from pathlib import Path as _Path

    cfg_path = _Path(local_dir) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type")
    except Exception:
        return None


def _load_gptq_causal_lm(model_id: str, load_kwargs: dict):
    """
    Load a GPTQ checkpoint.

    Path A: transformers + auto-gptq / optimum (1B Windows / older stacks).
    Path B: gptqmodel (Kaggle Python 3.12 — auto-gptq has no wheels).
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        _repair_gptq_g_idx_if_needed(model)
        return model
    except Exception as exc:
        print(
            f"HF GPTQ load failed ({type(exc).__name__}: {exc}). "
            "Falling back to gptqmodel..."
        )

    try:
        from gptqmodel import GPTQModel
    except ImportError as e:
        raise ImportError(
            "GPTQ load needs either auto-gptq+optimum or gptqmodel. "
            "On Kaggle Py3.12 install gptqmodel."
        ) from e

    from huggingface_hub import snapshot_download
    from pathlib import Path as _Path

    src = _Path(model_id)
    if src.is_dir():
        local_dir = str(src)
    else:
        local_dir = snapshot_download(model_id)
    _sanitize_gptq_configs_for_gptqmodel(local_dir)

    device = _device_from_map(load_kwargs.get("device_map"))
    # Marlin JIT often fails on T4 (sm_75) with current gptqmodel+torch; use Torch kernels.
    load_kw = {"device": device}
    try:
        from gptqmodel import BACKEND

        load_kw["backend"] = BACKEND.TORCH
        print("gptqmodel: forcing BACKEND.TORCH (Marlin JIT unreliable on T4)")
    except Exception:
        pass
    # Llama T4: fp16 compute so JuICE head matmuls match activations.
    # Gemma-2 GPTQ dumps are bf16; forcing float16 doubled RAM and SIGKILL'd Kaggle.
    if _gptq_config_model_type(local_dir) == "gemma2":
        print("gptqmodel: skipping torch_dtype for Gemma-2 (native dump dtype)")
    else:
        load_kw.setdefault("torch_dtype", torch.float16)
    loaded = GPTQModel.load(local_dir, **load_kw)
    # Prefer the underlying HF module so generate()/device_map helpers keep working.
    model = getattr(loaded, "model", None)
    if model is None:
        model = loaded
    _repair_gptq_g_idx_if_needed(model)
    return model


def load_model_and_tokenizer(
    precision: Precision,
    *,
    model_id_fp16: str = FP16_MODEL_ID,
    model_id_int4: str = GPTQ_MODEL_ID,
    quant_backend: QuantBackend = "gptq",
    device_map: DeviceMapArg = None,
    max_memory: dict | None = None,
    forbid_cpu_offload: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load exactly one precision. Caller must call unload_model() before loading another.

    quant_backend (INT4 only):
      - gptq: plan-primary path (GPTQ checkpoint via HF / auto-gptq)
      - bnb_nf4: local Windows smoke-only substitute (bitsandbytes NF4 on FP16 weights)

    device_map:
      None → {"": 0} on CUDA (1B default; avoids disk offload on 4GB).
      "auto" → shard across visible GPUs. Use this for 8B FP16 on Kaggle T4 x2.
      dict → passed through (e.g. {"": 0} to pin GPTQ on GPU 0).

    forbid_cpu_offload: with device_map="auto", cap CPU at 0 GiB so a too-big
    model errors instead of swapping. Pair with max_memory or gpu_max_memory().
    """
    resolved_map = resolve_device_map(
        device_map, precision=precision, quant_backend=quant_backend
    )
    load_kwargs: dict = {
        "device_map": resolved_map,
        "low_cpu_mem_usage": True,
    }
    if resolved_map == "auto":
        if max_memory is None:
            max_memory = gpu_max_memory(forbid_cpu_offload=forbid_cpu_offload)
        elif forbid_cpu_offload:
            max_memory = {**max_memory, "cpu": 0}
        load_kwargs["max_memory"] = max_memory

    if precision == "fp16":
        model_id = model_id_fp16
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            **load_kwargs,
        )
    elif precision == "int4":
        if quant_backend == "gptq":
            model_id = model_id_int4
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            except Exception as exc:
                is_gemma = "gemma" in str(model_id).lower()
                try:
                    is_gemma = is_gemma or _gptq_config_model_type(str(model_id)) == "gemma2"
                except Exception:
                    pass
                if not is_gemma:
                    raise
                print(
                    f"tokenizer from {model_id} failed ({type(exc).__name__}: {exc}); "
                    "falling back to google/gemma-2-9b"
                )
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Prefer HF+auto-gptq when available; on Kaggle Py3.12 use gptqmodel.
            model = _load_gptq_causal_lm(model_id, load_kwargs)
        elif quant_backend == "bnb_nf4":
            # LOCAL DEBUG ONLY — not interchangeable with GPTQ for Fix 1 / Stage 2.
            from transformers import BitsAndBytesConfig

            model_id = model_id_fp16
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                **load_kwargs,
            )
        else:
            raise ValueError(f"Unknown quant_backend: {quant_backend}")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model.eval()
    return model, tokenizer


def vram_mib() -> float | None:
    """Peak allocated MiB summed across all CUDA devices (not just the current one)."""
    if not torch.cuda.is_available():
        return None
    total = 0.0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.max_memory_allocated(i) / (1024**2)
    return total


def vram_per_device_mib() -> dict[int, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        i: torch.cuda.max_memory_allocated(i) / (1024**2)
        for i in range(torch.cuda.device_count())
    }


# Names used by mini_pilot / stage2 scripts.
FP16_MODEL_ID = FP16_MODEL_ID
GPTQ_MODEL_ID = GPTQ_MODEL_ID
unload_model = unload_model
load_model_and_tokenizer = load_model_and_tokenizer
