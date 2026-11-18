Kaggle Stage 2 JuICE FP16 upload pack
=====================================

Contents:
  common/           Python package (real JuICE dual-run)
  data/conflictqa_pilot250_popular.jsonl
  juice_repro/kaggle_stage2_fp16_8b.ipynb

How to use:
1. Create a Kaggle Dataset from this zip (or upload the zip as a dataset).
2. New Notebook -> GPU T4 x2 -> Internet ON -> attach this dataset.
3. Add secret HF_TOKEN; accept Llama-3.1-8B-Instruct license on HF.
4. File -> Import notebook -> juice_repro/kaggle_stage2_fp16_8b.ipynb
   OR open the .ipynb from the attached dataset and copy cells.
5. Run All (Cell 2 head ID takes ~6-9 hours).
