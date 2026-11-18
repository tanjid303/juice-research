# Kaggle — reverse transfer + native INT4 context K10

Cell A: native INT4 context K10 heads → FP16.
Cell B: same heads → N_CALIB=128 INT4 (in-precision).

**This zip has no weights.** Attach **ungabunga** output **346038437**.

## Run

1. Upload `kaggle_juice_stage2_int4_reverse_native_k10.zip` as a dataset
2. **File → Upload notebook** `kaggle_stage2_int4_reverse_native_k10.ipynb`
3. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
4. **Add Input → Notebook Output Files** → **ungabunga** version **346038437**
5. Save & Run All (~3–5 h)

Download:

```
/kaggle/working/juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_summary.json
/kaggle/working/juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_bootstrap.json
```

Locked: INT4 heads from `heads_int4_native_context_k10.json` (L7_H26 … L9_H0).
FP16 K10 (comparison only) from `heads_dir_context_k10.json`. Jaccard 0.33.
Baselines reused; α+=1, α−=0, split.json seed 42, bootstrap 1000 / seed 42.
