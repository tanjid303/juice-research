# Rebuild the lost N_CALIB=128 GPTQ checkpoint

Quantize **only**. Does not re-run transfer cells.

Output: `/kaggle/working/llama31_8b_base_gptq_int4_ncalib128/` (~4–6 GB)

## Run

1. Upload this zip as a Kaggle dataset
2. **File → Upload notebook** `kaggle_stage2_int4_ncalib128_quantize_only.ipynb`
3. GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN`
4. Accept the Llama-3.1-8B license on Hugging Face
5. Save & Run All (~45–90 min)
6. When it prints **SAVE VERSION NOW**, click Save Version immediately

Then open the native-clean-K2 notebook and **Add Input → Notebook Output Files → this Save Version**.
