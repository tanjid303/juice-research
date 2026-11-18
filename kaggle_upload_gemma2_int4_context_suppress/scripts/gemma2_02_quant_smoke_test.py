"""Sanity-check the existing Gemma-2-9B GPTQ N_CALIB=128 checkpoint.

Load BACKEND.TORCH (no requantize). Greedy completions on generic prompts and
JuICE coherent_conflict prompts. Decode strings, not token tensors.
"""
from __future__ import annotations

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


def _bootstrap_repo() -> Path:
    cands = list(Path("/kaggle/working").rglob("common/models.py"))
    cands += list(Path("/kaggle/input").rglob("common/models.py")) if Path("/kaggle/input").exists() else []
    assert cands, "Attach kaggle_juice_gemma2_int4_transfer_l27h6.zip (must contain common/)"

    def _ok(models_py: Path) -> bool:
        ja = models_py.parent / "juice_attn.py"
        if not ja.is_file():
            return False
        txt = ja.read_text(encoding="utf-8", errors="ignore")
        return "GetHeadsGemma2Attention" in txt and "dense_linear_weight_T" in txt

    cands.sort(key=lambda p: (_ok(p), "gemma2" in str(p).lower()), reverse=True)
    assert _ok(cands[0]), "common/juice_attn.py missing GetHeadsGemma2Attention / dense_linear_weight_T"
    repo = cands[0].parent.parent
    dst_common = Path("/kaggle/working/common")
    if dst_common.exists():
        shutil.rmtree(dst_common)
    shutil.copytree(repo / "common", dst_common)
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    print("REPO", repo)
    return repo


def _is_garbage(text: str, prompt: str) -> bool:
    body = text[len(prompt) :] if text.startswith(prompt) else text
    compact = "".join(body.split())
    if len(compact) < 3:
        return True
    if compact == compact[0] * len(compact):
        return True
    letters = sum(ch.isalpha() for ch in body)
    return letters < 2


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
            "huggingface-hub",
            "tqdm",
            "numpy",
            "pandas",
        ]
    )
    elapsed("pip install")

    import torch
    import transformers

    print("torch", torch.__version__, "transformers", transformers.__version__, "ngpu", torch.cuda.device_count())
    assert torch.cuda.is_available()

    repo = _bootstrap_repo()
    try:
        from huggingface_hub import login
        from kaggle_secrets import UserSecretsClient

        login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
        print("HF login ok")
    except Exception as e:
        print("HF login skipped:", type(e).__name__, e)

    from common.conflictqa import load_conflict_items
    from common.inference import greedy_generate
    from common.juice import conflict_eval_prompt
    from common.models import load_model_and_tokenizer, resolve_gemma2_ncalib128_checkpoint, unload_model

    OUT_DIR = Path("/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = resolve_gemma2_ncalib128_checkpoint()
    print("INT4_MODEL_ID", ckpt)
    meta_path = Path(ckpt) / "quant_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("quant_meta n_calib", meta.get("n_calib"), "bits", meta.get("bits"), "gs", meta.get("group_size"))
        assert int(meta.get("n_calib", 0)) == 128, meta.get("n_calib")

    print("Loading INT4 (BACKEND.TORCH, no requantize)", ckpt)
    model, tokenizer = load_model_and_tokenizer(
        "int4",
        model_id_int4=ckpt,
        device_map={"": 0},
    )
    print("model_type", getattr(model.config, "model_type", None))
    print("quantization_config", getattr(model.config, "quantization_config", None))
    assert getattr(model.config, "model_type", None) == "gemma2"
    elapsed("model loaded")

    prompts = [
        "The capital of France is",
        "Question: What is 2 + 2?\nAnswer:",
        "Once upon a time",
    ]
    data_hits = list(Path("/kaggle/working").rglob("conflictqa_pilot250_popular.jsonl"))
    data_hits += list(Path("/kaggle/input").rglob("conflictqa_pilot250_popular.jsonl")) if Path("/kaggle/input").exists() else []
    split_hits = list(Path("/kaggle/working").rglob("split.json"))
    split_hits += list(Path("/kaggle/input").rglob("split.json")) if Path("/kaggle/input").exists() else []
    split_hits = [p for p in split_hits if p.is_file()]
    split_hits.sort(key=lambda p: (0 if "gemma2" in str(p).lower() else 1, len(str(p))))
    if data_hits and split_hits:
        split = json.loads(split_hits[0].read_text(encoding="utf-8"))
        eval_ids = set(split["eval_ids"][:2])
        items = [it for it in load_conflict_items(data_hits[0]) if it.id in eval_ids]
        for it in items:
            prompts.append(conflict_eval_prompt(tokenizer, it, prompt_format="juice_coherent"))

    generations = []
    garbage = []
    for p in prompts:
        text = greedy_generate(model, tokenizer, p, max_new_tokens=16)
        generations.append({"prompt": p[:240], "out": text})
        print("GEN", repr(p[:80]), "->", repr(text))
        if _is_garbage(text, p):
            garbage.append(p[:80])

    payload = {
        "ok": not garbage,
        "checkpoint": ckpt,
        "n_prompts": len(prompts),
        "garbage": garbage,
        "generations": generations,
        "note": "JuICE-prompt smoke on existing N_CALIB=128 checkpoint. No requantize.",
    }
    (OUT_DIR / "decode_smoke_juice.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if garbage:
        raise AssertionError(f"garbage generations: {garbage}")
    print("SMOKE OK (decoded text, BACKEND.TORCH, JuICE prompts)")
    unload_model(model)
    elapsed("smoke done")
    print("REPO used", repo)


if __name__ == "__main__":
    main()
