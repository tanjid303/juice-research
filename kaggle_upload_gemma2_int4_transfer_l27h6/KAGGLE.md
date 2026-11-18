# Gemma-2-9B — INT4 transfer of L27_H6 (no requantize)

Weights are **not** in this zip. Attach the existing N_CALIB=128 checkpoint.

## Task 1 (priority)

Notebook: `kaggle_gemma2_int4_transfer_l27h6.ipynb`

1. GPU **T4 x2**, Internet ON, `HF_TOKEN`
2. Upload `kaggle_juice_gemma2_int4_transfer_l27h6.zip` as a dataset
3. **Add Input → Notebook Output Files** → gemma-quantize version with
   `gemma2_9b_base_gptq_int4_ncalib128/` (`config.json` + `*.safetensors`)
4. File → Upload the Task 1 notebook
5. Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.

Smoke (`gemma2_02`) then transfer (`gemma2_03`): L27_H6 K=1 + random-K1 (seed 42, exclude L27_H6).
Same 50 eval ids, `juice_coherent`, α+=1, α−=0, bootstrap 1000 / seed 42.

Download:

```
/kaggle/working/gemma2_int4_transfer_l27h6_summary.json
/kaggle/working/gemma2_int4_transfer_l27h6_bootstrap.json
```

Do **not** attach `gemma2_gptq_ncalib8`. Do **not** requantize.

## Task 2 (lower priority, FP16 only)

Notebook: `kaggle_gemma2_fp16_context_suppress.ipynb`

No quantized model. Same zip. New draft if Task 1 used the session.
Steers directional ctx K10 as H− (α+=0, α−=1).

```
/kaggle/working/gemma2_fp16_context_suppress_summary.json
/kaggle/working/gemma2_fp16_context_suppress_bootstrap.json
```
