# GEMMA2_05 — INT4 context K10 suppress transfer

No requantize. Attach the existing N_CALIB=128 checkpoint.

## Upload

**File → Upload notebook** `kaggle_gemma2_int4_context_suppress.ipynb`

Or paste `00_RUN_GEMMA2_05.py` as the only cell.

Do **not** upload `_make_*.py`. First print must be `THIS IS GEMMA2_05`.

## Attach

1. This zip as a dataset
2. **Add Input → Notebook Output Files** → gemma-quantize version with
   `gemma2_9b_base_gptq_int4_ncalib128/` (`config.json` + `*.safetensors`)

GPU T4 x2, Internet ON, HF_TOKEN. **New notebook.** Run All in the draft.
Quick Save when it prints `SAVE VERSION NOW`.

Scripts 01–04 are in `scripts/` for reference. This notebook only runs 05.

Random-K10 control reuses the Gemma FP16 Cell-3 enlarge draw (seed 42),
applied as H−. Not a new sample.

```
/kaggle/working/gemma2_int4_context_suppress_summary.json
/kaggle/working/gemma2_int4_context_suppress_bootstrap.json
```
