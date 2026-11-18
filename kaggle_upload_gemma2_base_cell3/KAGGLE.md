# Gemma-2-9B base Cell-3 FP16 apply

Does **not** use the INT4 checkpoint. Same 50 eval IDs as Gemma ID / Llama Stage 2.

| Condition | Heads |
|-----------|-------|
| steer_context_k10 | directional ctx top-10 (headline) |
| steer_parametric_k2_clean | L31_H7, L39_H6 |
| steer_parametric_k1_raw | L27_H6 (Llama L1_H4 analog) |
| random_context_k10 | seed 42, exclude identified set |

Protocol: `juice_coherent`, greedy, α+=1, α−=0, paired bootstrap 1000 / seed 42.

## Run

1. New notebook, GPU **T4 x2**, Internet ON, secret `HF_TOKEN`
2. Upload `kaggle_juice_gemma2_base_cell3.zip` as a dataset
3. File → Upload `kaggle_gemma2_base_cell3.ipynb`
4. Run All in the **draft** (~1.5–3 h). Do **not** Save Version first.
5. Quick Save when it prints `SAVE VERSION NOW`.

Need: `juice_stage2_gemma2_base_cell3/gemma2_base_cell3_summary.json` and `gemma2_base_cell3_bootstrap.json`.

Do **not** start GPTQ apply/transfer until the FP16 gate table is reported.
