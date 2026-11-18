# Kaggle — native INT4 clean K2 on N_CALIB=128

Steer `L16_H30` + `L17_H27` (native INT4 clean-filter pair, never steered) on the
existing `llama31_8b_base_gptq_int4_ncalib128` checkpoint.

**This zip has no weights.** Attach **ungabunga** output **346038437**.

## 1. Upload this zip as a dataset

File: `kaggle_juice_stage2_int4_native_clean_k2.zip`

## 2. Attach the N_CALIB=128 checkpoint (required)

In the notebook sidebar:

1. **Add Input → Notebook Output Files** (not Datasets)
2. Pick notebook **ungabunga**, version **346038437**
3. Confirm you see `llama31_8b_base_gptq_int4_ncalib128/` with `config.json` and `*.safetensors`

Do **not** attach `llama31_8b_base_gptq_int4` (that is N_CALIB=64).

## 3. Upload the notebook

**File → Upload notebook** → `kaggle_stage2_int4_native_clean_k2.ipynb` (zip root).

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Datasets | this zip **and** the N_CALIB=128 checkpoint |
| Secret | `HF_TOKEN` only if tokenizer files are missing |

## 4. Save & Run All

~1.5–3 h. Download:

```
/kaggle/working/juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json
/kaggle/working/juice_stage2_int4_ncalib128/int4_native_clean_k2_bootstrap.json
/kaggle/working/juice_stage2_int4_ncalib128/steer_int4_native_parametric_k2_clean.json
```

Locked: K=2, α+=1, α−=0, 50 ids in `directional_heads/split.json` (seed 42),
prompt `juice_coherent`, greedy 16 tokens, bootstrap 1000 / seed 42.
JuICE dual-run port (`common/juice.py`, `common/juice_attn.py`) is copied unchanged.
