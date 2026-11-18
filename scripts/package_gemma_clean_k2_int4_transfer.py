"""Package the INT4 transfer test for Gemma clean-K2 with matched random control as a Kaggle-ready zip + notebook.

Closes disclosed gap (A).

Assembles:
  - common/ (juice, models, scoring, stats_analysis, etc.)
  - data/ (conflictqa_pilot250_popular.jsonl)
  - directional_heads/ (split.json, heads_dir_parametric_k2_clean.json, heads_dir_context_k10.json, heads_dir_parametric_k1_raw.json, heads_random_k2.json)
  - scripts/ (run_gemma_clean_k2_int4_transfer.py)
  - README.md & KAGGLE.md
  - kaggle_gemma_clean_k2_int4_transfer.zip
  - kaggle_gemma_clean_k2_int4_transfer.ipynb
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "kaggle_upload_gemma_clean_k2_int4_transfer"
ZIP_PATH = ROOT / "kaggle_gemma_clean_k2_int4_transfer.zip"
NOTEBOOK_PATH = ROOT / "kaggle_gemma_clean_k2_int4_transfer.ipynb"

# Canonical Gemma source directory
GEMMA_CANONICAL_HEADS_DIR = ROOT / "juice_stage2_gemma2_base_id" / "directional_heads"


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int

    def key(self) -> str:
        return f"L{self.layer}_H{self.head}"


def head_keys(heads: Iterable[HeadRef]) -> list[str]:
    return [h.key() for h in heads]


def random_head_control(
    n_layers: int,
    n_heads: int,
    K: int,
    *,
    seed: int,
    exclude: Iterable[HeadRef] | None = None,
) -> list[HeadRef]:
    exclude_set = {h.key() for h in (exclude or [])}
    pool = [
        HeadRef(l, h)
        for l in range(n_layers)
        for h in range(n_heads)
        if HeadRef(l, h).key() not in exclude_set
    ]
    rng = random.Random(seed)
    return rng.sample(pool, k=min(K, len(pool)))


def build_random_k2_head_json(heads_dir: Path) -> Path:
    par_path = GEMMA_CANONICAL_HEADS_DIR / "heads_dir_parametric_k2_clean.json"
    ctx_path = GEMMA_CANONICAL_HEADS_DIR / "heads_dir_context_k10.json"
    raw_path = GEMMA_CANONICAL_HEADS_DIR / "heads_dir_parametric_k1_raw.json"

    assert par_path.is_file(), f"Canonical Gemma clean K2 file not found: {par_path}"
    assert ctx_path.is_file(), f"Canonical Gemma context K10 file not found: {ctx_path}"
    assert raw_path.is_file(), f"Canonical Gemma raw peak file not found: {raw_path}"

    par_data = json.loads(par_path.read_text(encoding="utf-8"))
    ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
    raw_data = json.loads(raw_path.read_text(encoding="utf-8"))

    par_heads = [HeadRef(h["layer"], h["head"]) for h in par_data.get("enlarge", [])]
    ctx_heads = [HeadRef(h["layer"], h["head"]) for h in ctx_data.get("enlarge", [])]
    raw_heads = [HeadRef(h["layer"], h["head"]) for h in raw_data.get("enlarge", [])]

    # CRITICAL SAFETY ASSERTION: Validates Gemma clean K2 heads match L31_H7, L39_H6 (NOT Llama heads)
    actual_par_keys = head_keys(par_heads)
    assert actual_par_keys == ["L31_H7", "L39_H6"], (
        f"CRITICAL SAFETY ASSERTION FAILED: Expected Gemma clean K2 heads ['L31_H7', 'L39_H6'], "
        f"but source {par_path} contains {actual_par_keys}. "
        "Packaging aborted to prevent deploying incorrect model heads."
    )
    assert len(ctx_heads) == 10, f"Expected 10 context heads, got {len(ctx_heads)}"
    assert head_keys(raw_heads) == ["L27_H6"], f"Expected raw peak L27_H6, got {head_keys(raw_heads)}"

    exclude = par_heads + ctx_heads + raw_heads

    # Gemma-2-9B has 42 layers, 16 query heads
    rand_heads = random_head_control(42, 16, 2, seed=42, exclude=exclude)
    assert head_keys(rand_heads) == ["L41_H11", "L7_H3"], (
        f"Expected random heads ['L41_H11', 'L7_H3'], got {head_keys(rand_heads)}"
    )

    payload = {
        "enlarge": [
            {"layer": h.layer, "head": h.head} for h in rand_heads
        ],
        "suppress": [],
        "metadata": {
            "protocol": "random_head_control",
            "seed": 42,
            "K": 2,
            "n_layers": 42,
            "n_heads": 16,
            "steer_target": "parametric",
            "excluded_heads": head_keys(exclude),
            "selected_heads": head_keys(rand_heads),
        }
    }
    target = heads_dir / "heads_random_k2.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def create_notebook(dst: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "gpu",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Gemma Clean-K2 INT4 Transfer Test & Matched Random Control\n",
                    "\n",
                    "**Closing Disclosed Gap (A)**:\n",
                    "This notebook applies Gemma-2-9B's locked parametric clean-K2 pair (`L31_H7, L39_H6`), identified on FP16, unchanged to the canonical GPTQ INT4 checkpoint (`gemma2_9b_base_gptq_int4_ncalib128`, $N_{\\mathrm{CALIB}}=128$).\n",
                    "\n",
                    "It also runs a matched random-head control ($K=2$, seed 42) for matched comparison:\n",
                    "- **Clean-K2 locked pair**: `L31_H7` and `L39_H6` (FP16 reference point: $-6\\ [-12, -1]$).\n",
                    "- **Random heads**: `L41_H11` and `L7_H3` (drawn with seed 42 from all 672 heads [42 layers $\\times$ 16 heads], excluding clean K2, context K10, and raw peak L27_H6).\n",
                    "- **Steering protocol**: Dual-run JuICE steering, $\\alpha_+=1.0, \\alpha_-=0.0$ (parametric enlarge direction), greedy decoding, `max_new_tokens=16`, `prompt_format=\"juice_coherent\"` on the preregistered seed-42 $n=100$ conflict items.\n",
                    "- **Precisions**: Sequentially runs (a) FP16 base (`google/gemma-2-9b`) and (b) canonical INT4 (`gemma2_9b_base_gptq_int4_ncalib128`), clearing VRAM between them.\n",
                    "- **Scoring**: Question-aware matcher + paired bootstrap (1000 resamples, seed 42) reporting both **gold-PAR-only $\\Delta$** and **inclusive $\\Delta$** extending Table 12.\n",
                    "\n",
                    "### Recommended Kaggle Settings\n",
                    "- **Accelerator**: GPU T4 x2 (Free tier)\n",
                    "- **Internet**: On (for Hugging Face model access)\n",
                    "- **Secrets**: Add `HF_TOKEN` with access to `google/gemma-2-9b` (accept Gemma license on Hugging Face)\n",
                    "- **Datasets attached**: \n",
                    "  1. `kaggle_gemma_clean_k2_int4_transfer.zip` (this package)\n",
                    "  2. `gemma2_9b_base_gptq_int4_ncalib128` (canonical GPTQ INT4 model directory from quantize notebook output)\n"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Setup environment, unpack package if attached as zip, authenticate Hugging Face, and inspect GPUs\n",
                    "import os\n",
                    "import sys\n",
                    "import zipfile\n",
                    "from pathlib import Path\n",
                    "import torch\n",
                    "\n",
                    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"Device count: {torch.cuda.device_count()}\")\n",
                    "    for d in range(torch.cuda.device_count()):\n",
                    "        print(f\"  Device {d}: {torch.cuda.get_device_name(d)} ({(torch.cuda.get_device_properties(d).total_memory / 1e9):.1f} GB)\")\n",
                    "\n",
                    "# Unpack dataset if zip is present in /kaggle/input\n",
                    "for p in Path(\"/kaggle/input\").rglob(\"*.zip\") if Path(\"/kaggle/input\").exists() else []:\n",
                    "    if \"gemma\" in str(p).lower() and (\"clean_k2\" in str(p).lower() or \"transfer\" in str(p).lower()):\n",
                    "        print(f\"Extracting {p} into /kaggle/working...\")\n",
                    "        try:\n",
                    "            with zipfile.ZipFile(p, 'r') as zf:\n",
                    "                zf.extractall(\"/kaggle/working\")\n",
                    "        except Exception as e:\n",
                    "            print(f\"Extract notice: {e}\")\n",
                    "\n",
                    "# Hugging Face Authentication via Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)\n",
                    "# Required because google/gemma-2-9b is a gated repository\n",
                    "try:\n",
                    "    from huggingface_hub import login\n",
                    "    from kaggle_secrets import UserSecretsClient\n",
                    "    sec = UserSecretsClient()\n",
                    "    hf_token = None\n",
                    "    for k in ['HF_TOKEN', 'hf_token', 'HUGGINGFACE_TOKEN', 'HUGGING_FACE_HUB_TOKEN']:\n",
                    "        try:\n",
                    "            hf_token = sec.get_secret(k)\n",
                    "            if hf_token:\n",
                    "                print(f\"[AUTH] Found secret key '{k}' in Kaggle Secrets.\")\n",
                    "                break\n",
                    "        except Exception:\n",
                    "            pass\n",
                    "    if hf_token:\n",
                    "        login(token=hf_token, add_to_git_credential=False)\n",
                    "        os.environ['HF_TOKEN'] = hf_token\n",
                    "        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token\n",
                    "        print('[AUTH] Successfully authenticated Hugging Face with Kaggle Secrets.')\n",
                    "    else:\n",
                    "        print('[AUTH] Warning: No HF_TOKEN secret found in Kaggle Secrets.')\n",
                    "except Exception as e:\n",
                    "    print(f'[AUTH] Kaggle Secrets login notice: {e}')\n",
                    "\n",
                    "# Install gptqmodel for Kaggle Python 3.12 (required for INT4 GPTQ load)\n",
                    "try:\n",
                    "    import gptqmodel\n",
                    "except ImportError:\n",
                    "    import subprocess\n",
                    "    print('Installing gptqmodel==2.2.0...')\n",
                    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gptqmodel==2.2.0', 'transformers==4.51.3', 'accelerate>=0.26.0,<2.0'])\n",
                    "    print('gptqmodel installed successfully.')\n",
                    "\n",
                    "working = Path(\"/kaggle/working\")\n",
                    "if str(working) not in sys.path:\n",
                    "    sys.path.insert(0, str(working))\n",
                    "\n",
                    "# Find run script\n",
                    "candidates = list(working.rglob(\"run_gemma_clean_k2_int4_transfer.py\"))\n",
                    "if Path(\"/kaggle/input\").exists():\n",
                    "    candidates += list(Path(\"/kaggle/input\").rglob(\"run_gemma_clean_k2_int4_transfer.py\"))\n",
                    "if not candidates:\n",
                    "    candidates = list(Path(\".\").rglob(\"run_gemma_clean_k2_int4_transfer.py\"))\n",
                    "\n",
                    "assert candidates, \"Could not find run_gemma_clean_k2_int4_transfer.py\"\n",
                    "SCRIPT_PATH = candidates[0]\n",
                    "print(f\"Run script resolved: {SCRIPT_PATH}\")\n"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Run Gemma Clean-K2 INT4 Transfer Test + Matched Random-K2 Control\n",
                    "import os\n",
                    "import subprocess\n",
                    "\n",
                    "cmd = [\n",
                    "    sys.executable, \"-u\", str(SCRIPT_PATH),\n",
                    "    \"--fp16-model\", \"google/gemma-2-9b\",\n",
                    "    \"--seed\", \"42\",\n",
                    "    \"--n-boot\", \"1000\",\n",
                    "]\n",
                    "\n",
                    "print(\"Starting experiment run...\")\n",
                    "env = os.environ.copy()\n",
                    "proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                    "for line in iter(proc.stdout.readline, ''):\n",
                    "    print(line, end='', flush=True)\n",
                    "proc.stdout.close()\n",
                    "ret = proc.wait()\n",
                    "assert ret == 0, f\"Experiment failed with returncode {ret}\"\n"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Display Table 12 Extension Results and inspect generated output files\n",
                    "import pandas as pd\n",
                    "from IPython.display import display, Markdown\n",
                    "\n",
                    "md_files = list(Path(\"/kaggle/working\").rglob(\"gemma_clean_k2_int4_transfer_n100_ncalib128.md\"))\n",
                    "csv_files = list(Path(\"/kaggle/working\").rglob(\"gemma_clean_k2_int4_transfer_n100_ncalib128.csv\"))\n",
                    "\n",
                    "if md_files:\n",
                    "    content = md_files[0].read_text(encoding=\"utf-8\")\n",
                    "    display(Markdown(\"### Table 12 Extension: Gemma Clean-K2 INT4 Transfer & Matched Controls\"))\n",
                    "    display(Markdown(content))\n",
                    "\n",
                    "if csv_files:\n",
                    "    df = pd.read_csv(csv_files[0])\n",
                    "    display(df)\n",
                    "\n",
                    "print(\"Done! You can download gemma_clean_k2_int4_transfer_n100_ncalib128.json, .md, and .csv from /kaggle/working.\")\n"
                ],
            },
        ],
    }
    dst.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Created notebook: {dst}")


def write_readme(dest: Path) -> None:
    text = """# Gemma Clean-K2 INT4 Transfer Test & Matched Random Control (NCALIB=128)

This package implements and runs the INT4 transfer test for Gemma-2-9B's locked parametric clean-K2 pair (`L31_H7, L39_H6`), closing disclosed gap (A), alongside a matched random-head control ($K=2$).

## 1. Experiment Specifications
- **Locked parametric pair**: `L31_H7, L39_H6` (identified on FP16; prior FP16 $n=100$ result: $-6\\ [-12, -1]$). Applied unchanged to INT4.
- **Random control**: $K=2$ attention heads drawn with `random_head_control` (seed 42) from all 672 heads (42 layers $\\times$ 16 heads).
  - Excluded from draw pool:
    - Clean K2 pair: `L31_H7, L39_H6`.
    - Context K10 set: `L40_H14, L17_H7, L24_H15, L22_H0, L32_H11, L25_H0, L3_H9, L8_H9, L7_H6, L29_H5`.
    - Raw peak: `L27_H6`.
  - Drawn heads: **`L41_H11`** and **`L7_H3`**.
- **JuICE steering**:
  - Decode-time dual-run steering ($\\alpha_+ = 1.0, \\alpha_- = 0.0$), greedy decoding, `max_new_tokens=16`, `prompt_format="juice_coherent"`.
- **Evaluation set**:
  - Preregistered seed-42 $n=100$ items from `data/conflictqa_pilot250_popular.jsonl` matching Gemma Stage 2 evaluation split.
- **Models**:
  - FP16 baseline: `google/gemma-2-9b` (reference point, reproduces $-6\\ [-12, -1]$).
  - Canonical INT4 checkpoint: `gemma2_9b_base_gptq_int4_ncalib128` ($N_{\\mathrm{CALIB}}=128$).
- **Scoring**:
  - Question-aware matcher (`label_conflict_response` with `question=gold_question`).
  - Inclusive $\\Delta$: Every baseline-context item that leaves context under steering.
  - Gold-PAR-only $\\Delta$: Leave-context counts only if steered completion mentions gold parametric entity (`_mentions_answer`).
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
"""
    (dest / "README.md").write_text(text, encoding="utf-8")
    (dest / "KAGGLE.md").write_text(text, encoding="utf-8")


def main() -> None:
    print(f"Building Kaggle upload bundle in {OUT_DIR}...")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # 1. Copy common
    common_files = [
        "__init__.py",
        "models.py",
        "juice.py",
        "juice_attn.py",
        "conflictqa.py",
        "scoring.py",
        "stats_analysis.py",
        "inference.py",
        "prompts.py",
    ]
    out_common = OUT_DIR / "common"
    out_common.mkdir(parents=True)
    for cf in common_files:
        src = ROOT / "common" / cf
        shutil.copy2(src, out_common / cf)

    # 2. Copy data
    out_data = OUT_DIR / "data"
    out_data.mkdir(parents=True)
    data_src = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
    assert data_src.is_file(), f"Dataset file not found: {data_src}"
    shutil.copy2(data_src, out_data / "conflictqa_pilot250_popular.jsonl")

    # 3. Copy directional_heads and build random head JSON
    out_heads = OUT_DIR / "directional_heads"
    out_heads.mkdir(parents=True)

    # Split: verify seed 42 and n_eval 100
    split_src = ROOT / "kaggle_upload_gemma_n100_verified" / "directional_heads" / "split.json"
    if not split_src.is_file():
        split_src = ROOT / "kaggle_upload_stage2_n100_headline" / "directional_heads" / "split.json"
    assert split_src.is_file(), f"split.json not found: {split_src}"
    split_data = json.loads(split_src.read_text(encoding="utf-8"))
    assert split_data.get("seed") == 42, f"Expected split seed 42, got {split_data.get('seed')}"
    assert split_data.get("n_eval") == 100, f"Expected n_eval 100, got {split_data.get('n_eval')}"
    shutil.copy2(split_src, out_heads / "split.json")

    # Copy explicitly from canonical Gemma directory
    par_src = GEMMA_CANONICAL_HEADS_DIR / "heads_dir_parametric_k2_clean.json"
    ctx_src = GEMMA_CANONICAL_HEADS_DIR / "heads_dir_context_k10.json"
    raw_src = GEMMA_CANONICAL_HEADS_DIR / "heads_dir_parametric_k1_raw.json"
    assert par_src.is_file(), f"Source clean K2 file not found: {par_src}"
    assert ctx_src.is_file(), f"Source context K10 file not found: {ctx_src}"
    assert raw_src.is_file(), f"Source raw peak file not found: {raw_src}"

    shutil.copy2(par_src, out_heads / "heads_dir_parametric_k2_clean.json")
    shutil.copy2(ctx_src, out_heads / "heads_dir_context_k10.json")
    shutil.copy2(raw_src, out_heads / "heads_dir_parametric_k1_raw.json")

    # Post-copy validation assertion: Gemma clean K2 must be L31_H7, L39_H6
    copied_par_data = json.loads((out_heads / "heads_dir_parametric_k2_clean.json").read_text(encoding="utf-8"))
    copied_par_keys = [f"L{h['layer']}_H{h['head']}" for h in copied_par_data.get("enlarge", [])]
    assert copied_par_keys == ["L31_H7", "L39_H6"], (
        f"Post-copy validation failed: Expected Gemma clean K2 ['L31_H7', 'L39_H6'], got {copied_par_keys}"
    )

    build_random_k2_head_json(out_heads)

    # 4. Copy scripts
    out_scripts = OUT_DIR / "scripts"
    out_scripts.mkdir(parents=True)
    runner_dest = out_scripts / "run_gemma_clean_k2_int4_transfer.py"
    shutil.copy2(ROOT / "scripts" / "run_gemma_clean_k2_int4_transfer.py", runner_dest)

    # Validate runner script AST
    import ast
    runner_code = runner_dest.read_text(encoding="utf-8")
    tree = ast.parse(runner_code)
    call_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_model_and_tokenizer":
            call_count += 1
            kw_names = [kw.arg for kw in node.keywords]
            if len(node.args) >= 1:
                arg0 = node.args[0]
                assert isinstance(arg0, ast.Constant) and arg0.value in ("fp16", "int4"), (
                    f"First positional argument to load_model_and_tokenizer must be 'fp16' or 'int4', got {ast.dump(arg0)}"
                )
                assert "precision" not in kw_names, (
                    "load_model_and_tokenizer got duplicate precision argument (passed both positionally and as keyword)!"
                )
    assert call_count >= 2, f"Expected at least 2 calls to load_model_and_tokenizer (fp16 & int4), found {call_count}"
    print(f"[RUNNER VERIFIED] Checked {call_count} load_model_and_tokenizer calls in runner script AST successfully.")

    # 5. Notebooks and documentation
    create_notebook(NOTEBOOK_PATH)
    shutil.copy2(NOTEBOOK_PATH, OUT_DIR / "kaggle_gemma_clean_k2_int4_transfer.ipynb")
    write_readme(OUT_DIR)

    # 6. Build zip
    print(f"Creating zip archive {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                rel = p.relative_to(OUT_DIR)
                zf.write(p, rel)

    zip_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"Zip created successfully: {ZIP_PATH} ({zip_mb:.2f} MB)")

    # 7. Post-build Zip Verification
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zip_names = zf.namelist()
        assert "directional_heads/heads_dir_parametric_k2_clean.json" in zip_names, (
            "Archive missing directional_heads/heads_dir_parametric_k2_clean.json!"
        )
        zip_json_raw = zf.read("directional_heads/heads_dir_parametric_k2_clean.json").decode("utf-8")
        zip_par_data = json.loads(zip_json_raw)
        zip_keys = [f"L{h['layer']}_H{h['head']}" for h in zip_par_data.get("enlarge", [])]
        assert zip_keys == ["L31_H7", "L39_H6"], (
            f"Zip verification failed: heads in zip are {zip_keys}, expected ['L31_H7', 'L39_H6']"
        )

        assert "directional_heads/heads_random_k2.json" in zip_names, (
            "Archive missing directional_heads/heads_random_k2.json!"
        )
        zip_rand_raw = zf.read("directional_heads/heads_random_k2.json").decode("utf-8")
        zip_rand_data = json.loads(zip_rand_raw)
        zip_rand_keys = [f"L{h['layer']}_H{h['head']}" for h in zip_rand_data.get("enlarge", [])]
        assert zip_rand_keys == ["L41_H11", "L7_H3"], (
            f"Zip verification failed: random heads in zip are {zip_rand_keys}, expected ['L41_H11', 'L7_H3']"
        )

        assert "directional_heads/split.json" in zip_names, "Archive missing split.json!"
        zip_split = json.loads(zf.read("directional_heads/split.json").decode("utf-8"))
        assert zip_split.get("n_eval") == 100, f"Expected n_eval=100 in zip split.json, got {zip_split.get('n_eval')}"

    print(f"[PACKAGE VERIFIED] Zip archive contains valid Gemma clean K2 heads: {zip_keys}")
    print(f"[PACKAGE VERIFIED] Zip archive contains valid Gemma random K2 heads: {zip_rand_keys}")
    print(f"[PACKAGE VERIFIED] Zip archive contains valid n=100 split.")


if __name__ == "__main__":
    main()
