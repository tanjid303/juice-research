# Local environment notes (3050 Ti debug tier)

## peft pin for auto-gptq

`auto-gptq==0.7.1` imports `PEFT_TYPE_TO_MODEL_MAPPING` from `peft.peft_model`.
That symbol was removed in newer peft (e.g. 0.20). Keep:

```
peft==0.10.0
```

in `requirements.txt`. Without this pin, GPTQ INT4 load fails on Windows with:

```
ImportError: cannot import name 'PEFT_TYPE_TO_MODEL_MAPPING' from 'peft.peft_model'
```

## Quantization policy

- **Paper / A4500 / Fix 1 / Stage 2:** GPTQ-INT4 only (per PROJECT_BRIEF_FULL.md).
- **3050 Ti Week 0 smoke / mini-pilot:** prefer `--quant_backend gptq` (requires `peft==0.10.0`).
- Between FP16 and INT4, scripts must fully unload (`model.to("cpu")` + `empty_cache`) or 4GB will
  disk-offload the GPTQ load.
- **Windows fallback:** if GPTQ still fails, use `--quant_backend bnb_nf4` for pipeline smoke only
  (bitsandbytes NF4 on the FP16 checkpoint — **not** GPTQ). Results log `quant_backend`; never mix
  bnb_nf4 into the main GPTQ table.

```powershell
pip install bitsandbytes
python mini_pilot/run_1b_fp16_vs_int4.py --n_items 20 --quant_backend bnb_nf4 --data data/conflictqa_smoke.jsonl
```

## GPTQ checkpoint quality (fbaldassarri AutoRound→GPTQ export)

Checked 2026-08-24 on 3050 Ti (`mini_pilot/check_gptq_quality.py`):

| Check | Result |
|-------|--------|
| Config | bits=4, group_size=128, sym=True, desc_act=False — matches the repo name |
| Origin | AutoRound (`autoround_version` in original config) exported in GPTQ layout — not vanilla GPTQ |
| Tensors | `qweight`/`qzeros`/`scales` present and correctly shaped; real QuantLinear, not FP16 fallback |
| Unused `*.bias` | AutoRound dumped biases; Llama-3.2 has `attention_bias=false` / `mlp_bias=false` — correctly ignored |
| `g_idx` newly initialized | Export omitted `g_idx` (normal when `desc_act=False`). Loader left zeros; **cuda-old** dequant uses `group_size`, not `g_idx`, so zeros were harmless. `common/models.py` now rewrites sequential `g_idx = i // group_size` after load |
| Generations | Coherent factual English (Paris, 100°C, Au, Jupiter, …) |
| PPL (154 toks) | FP16 8.86 → INT4 9.10 (**+2.6%**) — well inside normal INT4 range |

**Trust for 1B pipeline / n=50 smoke:** yes, enough to scale the mini-pilot pipeline.

**Do not treat as final paper GPTQ:** for A4500 Stage 1/2, prefer a self-run GPTQ (or a known-vanilla GPTQ export) so the algorithm matches the locked “GPTQ only” policy, not AutoRound-in-GPTQ-clothing.
