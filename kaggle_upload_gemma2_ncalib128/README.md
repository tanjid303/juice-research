# Gemma-2-9B N_CALIB=128 GPTQ (quantize only)

Canonical INT4 for Gemma Stage 2. Not the N_CALIB=8 smoke.

Output: `/kaggle/working/gemma2_9b_base_gptq_int4_ncalib128/` (~5–7 GB)

## Run

1. **New** notebook — do not reuse the smoke notebook
2. Upload `kaggle_juice_gemma2_ncalib128.zip` as a dataset
3. File → Upload notebook `kaggle_gemma2_ncalib128_quantize.ipynb`
4. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
5. Accept google/gemma-2-9b
6. **Run All in the draft** (~60–120 min). Do **not** click Save Version first.
7. When it prints **SAVE VERSION NOW**, **Quick Save** (not Save & Run All).
   Failed Save & Run All wipes the checkpoint (Gemma Version 2).
