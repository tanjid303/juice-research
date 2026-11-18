# Kaggle — Native INT4 ID + random-K + Jaccard

Weights are **not** in this zip. Attach the quantized model separately.

## 1. Upload this zip as a dataset
`kaggle_juice_stage2_int4_native_id.zip`

## 2. Attach the GPTQ checkpoint

Add Input → Notebook Output Files → the quantize notebook Save Version
that contains `llama31_8b_base_gptq_int4/`.

## 3. New notebook
Upload `juice_repro/kaggle_stage2_int4_native_id.ipynb`.

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Datasets | this zip **and** the GPTQ checkpoint |
| Secret | `HF_TOKEN` if tokenizer files are missing |
| Runtime | ~3–6 h (ID is 2–4 h; random-K is extra) |

## 4. Outputs

```
/kaggle/working/juice_stage2_int4_native_id/int4_native_id_summary.json
/kaggle/working/juice_stage2_int4_native_id/jaccard_bootstrap.json
/kaggle/working/juice_stage2_int4_native_id/int4_random_k_summary.json
/kaggle/working/juice_stage2_int4_native_id/id_per_item_scores.json
```

If auto-detect fails, set in cell 2:

```python
INT4_MODEL_ID_OVERRIDE = "/kaggle/input/<slug>/llama31_8b_base_gptq_int4"
```
