# Kaggle — Stage 2 FP16→INT4 transfer (base)

Weights are **not** in this zip. Attach the quantized model separately.

## 1. Upload this zip as a dataset
`kaggle_juice_stage2_int4_transfer.zip`

## 2. Attach the GPTQ checkpoint (required — this zip has no weights)

In the transfer notebook sidebar:

1. **Add Input → Notebook Output Files** (not “Datasets”)
2. Pick the **quantize** notebook’s **Save Version** (the run that printed `Done. Download: /kaggle/working/llama31_8b_base_gptq_int4`)
3. Confirm you see a folder with `config.json` and `*.safetensors` (~4–6 GB)

The code zip (`1234567890` / `kaggle_juice_stage2_int4_transfer`) is not the model.

If auto-detect still fails, set in cell 2:

```python
INT4_MODEL_ID_OVERRIDE = "/kaggle/input/<slug>/llama31_8b_base_gptq_int4"
```

## 3. New notebook
Upload `juice_repro/kaggle_stage2_int4_transfer.ipynb`.

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Datasets | this zip **and** the GPTQ checkpoint |
| Secret | `HF_TOKEN` only if tokenizer files are missing |

## 4. Run
Save & Run All. Output:

```
/kaggle/working/juice_stage2_int4_transfer/int4_transfer_summary.json
```

Gate: context K10/K5 and parametric clean K2 must keep the **same sign** as FP16
(+10 / +6 / −8 pp) with coherence ≥ 0.90.
