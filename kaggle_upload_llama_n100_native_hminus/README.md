# Llama n=100 remaining Figure 3 cells

Attach this dataset and the existing Llama N_CALIB=128 GPTQ checkpoint. Run `kaggle_llama_n100_native_hminus.ipynb` on GPU T4 x2.

Runs native INT4 context K10 enlarge and FP16 context K10 H-minus suppression on the exact n=100 seed-42 IDs. Bootstrap: 1000 draws, seed 42.

Output: `/kaggle/working/llama3_int4_n100_remaining/`.
