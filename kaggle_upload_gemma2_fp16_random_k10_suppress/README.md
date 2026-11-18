# GEMMA2_06 — FP16 random-K10 suppress

**No INT4 checkpoint.** New notebook. Do **not** upload `_make_*.py`.

File → Upload `kaggle_gemma2_fp16_random_k10_suppress.ipynb`
or paste `00_RUN_GEMMA2_06.py` as the only cell.

First print must be `THIS IS GEMMA2_06`.

GPU T4 x2, Internet ON, HF_TOKEN. Attach this zip. Run All in the draft.

Same 10 heads as the INT4 suppress random control (Cell-3 enlarge draw, seed 42),
applied as H− (α+=0, α−=1). Same 50 ids, juice_coherent, bootstrap 1000 / seed 42.

```
/kaggle/working/gemma2_fp16_random_k10_suppress_summary.json
/kaggle/working/gemma2_fp16_random_k10_suppress_bootstrap.json
```
