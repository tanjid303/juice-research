# Kaggle — Quantize Llama-3.1-8B base → GPTQ INT4

## 1. Upload dataset
- Upload **`kaggle_quantize_llama31_8b_base.zip`** as a new Kaggle Dataset.

## 2. New notebook
- **File → New Notebook → Upload Notebook**
- Pick `juice_repro/kaggle_quantize_llama31_8b_base_gptq.ipynb` from the dataset  
  OR copy notebook from dataset into a fresh notebook (delete old auto-gptq notebook).

## 3. Notebook settings
| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Secret | `HF_TOKEN` |

## 4. Hugging Face
Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B

## 5. Run
- **Save & Run All** (not Run cell-by-cell on a stale kernel)
- First run: `N_CALIB = 64` in cell 2 (smoke, ~20–30 min)
- Then: `N_CALIB = 128`, then `512` for final checkpoint
- **Do not skip truncation** — cell 4 tokenizes with `MAX_SEQ_LEN = 512` (full C4 docs OOM)

## 6. Output
```
/kaggle/working/llama31_8b_base_gptq_int4/
```
**Save Version** → download → re-upload as dataset for INT4 JuICE transfer.

## Tooling
- **`gptqmodel==2.2.0`** + **`transformers==4.51.3`** (not 4.56+/5.x — those break RoPE on layer 1)
- Does **NOT** use auto-gptq (no Py3.12 wheel)
- **Restart kernel** after upload; do not reuse a mid-quantize crash

## Plan B
Community checkpoint: `ModelCloud/Meta-Llama-3.1-8B-gptq-4bit`
