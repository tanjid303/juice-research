# Llama Clean-K2 Matched Random-K2 Control (FP16 & INT4 NCALIB=128)

This package implements and runs the matched random-head control ($K=2$) for the Llama clean-K2 headline finding on both FP16 (`meta-llama/Llama-3.1-8B`) and canonical INT4 (`llama31_8b_base_gptq_int4_ncalib128`).

## 1. Protocol Specifications
- **Head selection**: $K=2$ attention heads selected using `random_head_control` with seed 42.
  - Excluded from draw pool:
    - Parametric clean K2 headline heads: `L20_H14`, `L26_H15`.
    - Context K10 heads: `L4_H9`, `L9_H0`, `L1_H11`, `L4_H5`, `L19_H27`, `L17_H12`, `L7_H26`, `L14_H23`, `L3_H22`, `L31_H20`.
  - Drawn heads: **`L20_H24`** and **`L3_H19`**.
- **JuICE steering**:
  - Decode-time dual-run steering ($lpha_+ = 1.0, lpha_- = 0.0$), greedy decoding, `max_new_tokens=16`.
- **Evaluation set**:
  - Seed-42 $n=100$ items from `data/conflictqa_pilot250_popular.jsonl` matching Llama Stage 2 diagnostic split.
- **Scoring**:
  - Question-aware matcher (`label_conflict_response` with `question=gold_question`).
  - Inclusive $\Delta$: Every baseline-context item that leaves context under steering.
  - Gold-PAR-only $\Delta$: Leave-context counts only if steered completion names the gold parametric entity (`_mentions_answer`).
  - Paired bootstrap: 1000 resamples, seed 42, percentile 95% CI.
- **Output naming**:
  - `llama_clean_k2_random_control_n100_ncalib128.json`
  - `llama_clean_k2_random_control_n100_ncalib128.md`
  - `llama_clean_k2_random_control_n100_ncalib128.csv`

## 2. Running on Kaggle (T4 x 2)
1. Upload `kaggle_llama_clean_k2_random_control.zip` as a dataset.
2. Attach `llama31_8b_base_gptq_int4_ncalib128` as a dataset (or provide checkpoint in working directory).
3. Open `kaggle_llama_clean_k2_random_control.ipynb`.
4. Enable **GPU T4 x2**, enable **Internet**, add Hugging Face token secret (`HF_TOKEN`) if gating requires it.
5. Click **Run All**.
