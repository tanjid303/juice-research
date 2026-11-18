# Gemma Clean-K2 INT4 Transfer Test & Matched Random Control (NCALIB=128)

This package implements and runs the INT4 transfer test for Gemma-2-9B's locked parametric clean-K2 pair (`L31_H7, L39_H6`), closing disclosed gap (A), alongside a matched random-head control ($K=2$).

## 1. Experiment Specifications
- **Locked parametric pair**: `L31_H7, L39_H6` (identified on FP16; prior FP16 $n=100$ result: $-6\ [-12, -1]$). Applied unchanged to INT4.
- **Random control**: $K=2$ attention heads drawn with `random_head_control` (seed 42) from all 672 heads (42 layers $\times$ 16 heads).
  - Excluded from draw pool:
    - Clean K2 pair: `L31_H7, L39_H6`.
    - Context K10 set: `L40_H14, L17_H7, L24_H15, L22_H0, L32_H11, L25_H0, L3_H9, L8_H9, L7_H6, L29_H5`.
    - Raw peak: `L27_H6`.
  - Drawn heads: **`L41_H11`** and **`L7_H3`**.
- **JuICE steering**:
  - Decode-time dual-run steering ($\alpha_+ = 1.0, \alpha_- = 0.0$), greedy decoding, `max_new_tokens=16`, `prompt_format="juice_coherent"`.
- **Evaluation set**:
  - Preregistered seed-42 $n=100$ items from `data/conflictqa_pilot250_popular.jsonl` matching Gemma Stage 2 evaluation split.
- **Models**:
  - FP16 baseline: `google/gemma-2-9b` (reference point, reproduces $-6\ [-12, -1]$).
  - Canonical INT4 checkpoint: `gemma2_9b_base_gptq_int4_ncalib128` ($N_{\mathrm{CALIB}}=128$).
- **Scoring**:
  - Question-aware matcher (`label_conflict_response` with `question=gold_question`).
  - Inclusive $\Delta$: Every baseline-context item that leaves context under steering.
  - Gold-PAR-only $\Delta$: Leave-context counts only if steered completion mentions gold parametric entity (`_mentions_answer`).
  - Paired bootstrap: 1000 resamples, seed 42, percentile 95% CI.
- **Output naming**:
  - `gemma_clean_k2_int4_transfer_n100_ncalib128.json`
  - `gemma_clean_k2_int4_transfer_n100_ncalib128.md`
  - `gemma_clean_k2_int4_transfer_n100_ncalib128.csv`

## 2. Running on Kaggle (T4 x 2)
1. Upload `kaggle_gemma_clean_k2_int4_transfer.zip` as a dataset.
2. Attach `gemma2_9b_base_gptq_int4_ncalib128` as a dataset (from quantize notebook output).
3. Open `kaggle_gemma_clean_k2_int4_transfer.ipynb`.
4. Enable **GPU T4 x2**, enable **Internet**, add Hugging Face token secret (`HF_TOKEN`) with access to `google/gemma-2-9b`.
5. Click **Run All**.
