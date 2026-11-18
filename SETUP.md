# Environment setup (3050 Ti / 4 GB VRAM debug tier)

## Python environment

```powershell
cd D:\tt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# Install CUDA PyTorch first if the plain pip torch wheel is CPU-only on your machine:
# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Windows: `fbgemm.dll` / WinError 126

Torch 2.4.0 Windows wheels import `libomp140.x86_64.dll`, which is not shipped. After a successful `pip install torch`, if `import torch` fails with that error, copy the bundled Intel OpenMP DLL next to it:

```powershell
$lib = ".\.venv\Lib\site-packages\torch\lib"
Copy-Item "$lib\libiomp5md.dll" "$lib\libomp140.x86_64.dll"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do **not** pull random copies of that DLL from DLL-download sites. Re-apply this copy if you reinstall torch.

Log in to Hugging Face before pulling gated Llama weights:

```powershell
huggingface-cli login
```

## VRAM monitoring (watch for OOM)

Keep a second terminal open while any script runs:

```powershell
# One-shot snapshot
nvidia-smi

# Refresh every 2 seconds (PowerShell)
while ($true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 2 }
```

What to watch:

| Signal | Likely cause | Smallest fix |
|--------|--------------|--------------|
| `Memory-Usage` near 4096 MiB before generation | Model still loaded + KV cache | `--max_new_tokens 16`, shorter prompts |
| Spike at model swap | Forgotten cleanup | Scripts call `unload_model()`; if hacking interactively, `del model; torch.cuda.empty_cache()` |
| OOM during JuICE dual-run | Two forward traces + hooks | `--n_items 20`, `--bootstrap_draws 3`, `--K 3` |
| OOM on GPTQ load | FP16 + INT4 both resident | Never load both; mini-pilot loads one precision at a time |

**Rule for 4 GB:** batch size 1, one model precision at a time, start with `--n_items 50` (or 20 for mechanistic scripts), scale up only after a clean run.

## Quick smoke order

1. `python juice_repro/their_setup_smoke.py --n_items 5`
2. `python mini_pilot/run_1b_fp16_vs_int4.py --n_items 20`
3. `python stage2_mechanistic/identify_heads.py --n_items 20 --bootstrap_draws 3`

Lab A4500 (20 GB) is the primary machine for full mini-pilot / Fix 1 numbers; this laptop validates the pipeline only.
