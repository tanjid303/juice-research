# Gemma-2-9B JuICE smoke (Kaggle T4 x2)

FP16 wrap + dual-run, then 1-layer / 8-sample GPTQ decode check.

**Do not** start N_CALIB=128 until GPTQ generations are real words.
**Do not** use `/kaggle/working/gemma2_gptq_tiny_layer0` as a Stage 2 checkpoint.

## Run

1. Upload `kaggle_juice_gemma2_smoke.zip` as a Kaggle dataset
2. **File → Upload notebook** `kaggle_gemma2_juice_smoke.ipynb`
3. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
4. Accept [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) on Hugging Face
5. Save & Run All (~20–40 min)

Pass: config 256 / 16 / 8 / 50, 42 wrapped layers, shape `[B, S, 3584]`, GPTQ gens not garbage.
Fail (garbage decode or o_proj wrap): pivot to Qwen2.5-7B.
