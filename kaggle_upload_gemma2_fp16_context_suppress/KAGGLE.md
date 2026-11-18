# TASK 2 — Gemma-2-9B FP16 context K10 suppress

**One notebook only.** File → Upload `kaggle_gemma2_fp16_context_suppress.ipynb`.
Do not upload `_make_*.py`.

No INT4 checkpoint. New Kaggle notebook, GPU T4 x2, Internet ON, HF_TOKEN.

1. Upload `kaggle_juice_gemma2_fp16_context_suppress.zip` as a dataset
   (or reuse the Task 1 zip dataset if it already has `gemma2_04_fp16_context_suppress_check.py`)
2. File → Upload `kaggle_gemma2_fp16_context_suppress.ipynb`
3. First code cell must print: `THIS IS TASK 2 — FP16 ctx K10 suppress`
4. Run All in the **draft**. Quick Save when it prints `SAVE VERSION NOW`.

α+=0, α−=1, enlarge empty, same 50 eval ids, juice_coherent, bootstrap 1000 / seed 42.

```
/kaggle/working/gemma2_fp16_context_suppress_summary.json
/kaggle/working/gemma2_fp16_context_suppress_bootstrap.json
```
