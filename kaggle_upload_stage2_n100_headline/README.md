# Stage 2 n=100 headline Kaggle package

This package runs only the preregistered n=100 evaluation set:
`random.Random(42)` over the full 250-item pool, with positions `25:125`
as the 100 evaluation IDs. It does not run n=150 or the repair suite.

## Four notebooks

Run each notebook in a fresh Kaggle draft with GPU T4 x2, Internet enabled,
and `HF_TOKEN` configured:

1. `juice_repro/kaggle_n100_llama_fp16.ipynb`
   - Llama FP16 baseline, context K10, clean parametric K2, random K10.
2. `juice_repro/kaggle_n100_llama_int4.ipynb`
   - Llama N_CALIB=128 INT4 baseline, context K10 transfer, random K10,
     clean parametric K2 transfer.
3. `juice_repro/kaggle_n100_gemma_fp16.ipynb`
   - Gemma FP16 baseline and raw L27_H6.
4. `juice_repro/kaggle_n100_gemma_int4.ipynb`
   - Gemma N_CALIB=128 INT4 baseline, raw L27_H6 transfer, random K1.

The INT4 notebooks require the existing N_CALIB=128 checkpoint as an
additional Notebook Output input. They do not requantize.

Every runner uses `juice_coherent`, greedy decoding, 16 new tokens,
alpha+=1, alpha-=0, paired bootstrap with 1000 draws and seed 42.

Download the JSON files from:

- `/kaggle/working/juice_stage2_llama_base_n100/`
- `/kaggle/working/juice_stage2_int4_ncalib128_n100/`
- `/kaggle/working/juice_stage2_gemma_base_n100/`
- `/kaggle/working/juice_stage2_gemma2_int4_transfer_l27h6_n100/`

Do not start n=150 until the n=100 outputs have been reviewed.
