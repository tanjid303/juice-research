# Llama n=100 INT4 parametric diagnostics

Attach this ZIP as a Kaggle dataset and attach the existing Llama-3.1-8B base GPTQ N_CALIB=128 checkpoint. Upload `kaggle_llama_n100_parametric_diagnostics.ipynb` and run on GPU T4 x2.

Runs: noisy K2/K3 controls, clean-pair alpha grid (0.5/1/2/3/5), native pair L16_H30+L17_H27, and raw L1_H4. All use n=100, seed 42, 1000 paired bootstrap draws, and alpha-minus=0.

Output folder: `/kaggle/working/llama3_int4_n100_diagnostics/`.
