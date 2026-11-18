# Kaggle — INT4 parametric recovery (L1_H4 + alpha grid + suppress ctx K10)

## Attach
1. This zip as a dataset
2. `quantizing-output` (GPTQ weights)

## Notebook
Upload `juice_repro/kaggle_stage2_int4_parametric_fixes.ipynb`

GPU T4 x2, Internet ON. ~40–70 min. **Do not** re-run native ID.

GPU order after baseline: L1_H4 (K=1) → clean-pair alpha grid → suppress ctx K10.

Output: `/kaggle/working/juice_stage2_int4_parametric_fixes/int4_parametric_fixes_summary.json`
Look at `l1_h4` first: negative delta with CI excluding 0 would mean parametric steering
works on INT4 via the raw-ID peak (L1_H4), not the directional clean pair.
