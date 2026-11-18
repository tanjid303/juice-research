# Gemma-2-9B n=100 expansion

Headline FP16/INT4 n=100 is already complete and included under `headline_results/`.
Do **not** rerun those cells.

## H- battery (this is the remaining work)

1. New Kaggle GPU T4 x2 draft. Factory reset if a previous INT4 session is still live.
2. Attach this ZIP, the Gemma N_CALIB=128 checkpoint, Internet, `HF_TOKEN`.
3. Upload and run `gemma_n100_hminus_battery.ipynb` **only**.
4. Each H- runner is a **subprocess**. The CUDA context dies before the next load.
5. Download `/kaggle/working/gemma_n100_fp16_hminus/`, `gemma_n100_fp16_random_hminus/`,
   `gemma_n100_int4_hminus/`, `gemma_n100_raw_results.json`, and
   `gemma_n100_hminus_specificity_did.json`.

Locked random-K10 (must appear in every random-control log):
L41_H11, L7_H3, L1_H9, L17_H13, L15_H13, L14_H7, L9_H1, L6_H9, L35_H9, L5_H10

Protocol: seed 42, 100 eval IDs, 1000 paired bootstrap draws, juice_coherent, 16 tokens.
Do not edit the paper until those five outputs are reviewed.
