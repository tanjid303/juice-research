"""Package the matched random-K2 control for Llama clean-K2 as a Kaggle-ready zip + notebook.

Assembles:
  - common/ (juice, models, scoring, stats_analysis, etc.)
  - data/ (conflictqa_pilot250_popular.jsonl)
  - directional_heads/ (split.json, heads_dir_parametric_k2_clean.json, heads_dir_context_k10.json, heads_random_k2.json)
  - scripts/ (run_llama_clean_k2_random_control.py)
  - README.md & KAGGLE.md
  - kaggle_llama_clean_k2_random_control.zip
  - kaggle_llama_clean_k2_random_control.ipynb
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "kaggle_upload_llama_clean_k2_random_control"
ZIP_PATH = ROOT / "kaggle_llama_clean_k2_random_control.zip"
NOTEBOOK_PATH = ROOT / "kaggle_llama_clean_k2_random_control.ipynb"

# Canonical, non-ambiguous Llama source directory
LLAMA_CANONICAL_HEADS_DIR = ROOT / "juice_stage2_base_id" / "directional_heads"


from dataclasses import dataclass
import random
from typing import Iterable


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
    par_path = LLAMA_CANONICAL_HEADS_DIR / "heads_dir_parametric_k2_clean.json"
    ctx_path = LLAMA_CANONICAL_HEADS_DIR / "heads_dir_context_k10.json"

    assert par_path.is_file(), f"Canonical Llama clean K2 file not found: {par_path}"
    assert ctx_path.is_file(), f"Canonical Llama context K10 file not found: {ctx_path}"

    par_data = json.loads(par_path.read_text(encoding="utf-8"))
    ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))

    par_heads = [HeadRef(h["layer"], h["head"]) for h in par_data.get("enlarge", [])]
    ctx_heads = [HeadRef(h["layer"], h["head"]) for h in ctx_data.get("enlarge", [])]

    # SAFETY ASSERTION: Validates Llama clean K2 heads match L20_H14, L26_H15 before zip build
    actual_par_keys = head_keys(par_heads)
    assert actual_par_keys == ["L20_H14", "L26_H15"], (
        f"CRITICAL SAFETY ASSERTION FAILED: Expected Llama clean K2 heads ['L20_H14', 'L26_H15'], "
        f"but source {par_path} contains {actual_par_keys}. "
        "Packaging aborted to prevent deploying incorrect model heads."
    )
    assert len(ctx_heads) == 10, f"Expected 10 context heads, got {len(ctx_heads)}"

    exclude = par_heads + ctx_heads

    rand_heads = random_head_control(32, 32, 2, seed=42, exclude=exclude)
    assert head_keys(rand_heads) == ["L20_H24", "L3_H19"]

    payload = {
        "enlarge": [
            {"layer": h.layer, "head": h.head} for h in rand_heads
        ],
        "suppress": [],
        "metadata": {
            "protocol": "random_head_control",
            "seed": 42,
            "K": 2,
            "n_layers": 32,
            "n_heads": 32,
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
                    "# Llama Clean-K2 Matched Random-K Control (FP16 & INT4 NCALIB=128)\n",
                    "\n",
                    "This notebook runs the matched random attention-head pair control ($K=2$, seed 42) for the Llama clean-K2 headline finding:\n",
                    "- **Random heads**: `L20_H24` and `L3_H19` (drawn with seed 42 from all 1024 heads, excluding the identified clean K2 pair `L20_H14`, `L26_H15` and the 10 context K10 heads).\n",
                    "- **Steering protocol**: Decode-time JuICE steering, $\\alpha_+=1.0, \\alpha_-=0.0$ (parametric enlarge direction), greedy decoding, `max_new_tokens=16`, on the preregistered $n=100$ conflict items.\n",
                    "- **Precisions**: Sequentially runs (a) FP16 base (`meta-llama/Llama-3.1-8B`) and (b) canonical INT4 (`llama31_8b_base_gptq_int4_ncalib128`).\n",
                    "- **Scoring**: Question-aware matcher + paired bootstrap (1000 resamples, seed 42) reporting both **inclusive $\\Delta$** and **gold-PAR-only $\\Delta$** matching Table 8.\n",
                    "\n",
                    "### Recommended Kaggle Settings\n",
                    "- **Accelerator**: GPU T4 x2 (Free tier)\n",
                    "- **Internet**: On (for downloading FP16 weights / Hugging Face access)\n",
                    "- **Secrets**: Add `HF_TOKEN` with read access to `meta-llama/Llama-3.1-8B`\n",
                    "- **Datasets attached**: \n",
                    "  1. `kaggle_llama_clean_k2_random_control.zip` (this package)\n",
                    "  2. `llama31_8b_base_gptq_int4_ncalib128` (canonical GPTQ INT4 model directory)\n"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Setup environment, unpack package if attached as zip, and inspect GPUs\n",
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
                    "    if \"random_control\" in str(p) or \"llama\" in str(p):\n",
                    "        print(f\"Extracting {p} into /kaggle/working...\")\n",
                    "        try:\n",
                    "            with zipfile.ZipFile(p, 'r') as zf:\n",
                    "                zf.extractall(\"/kaggle/working\")\n",
                    "        except Exception as e:\n",
                    "            print(f\"Extract notice: {e}\")\n",
                    "\n",
                    "# HF Authentication via Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)\n",
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
                    "candidates = list(working.rglob(\"run_llama_clean_k2_random_control.py\"))\n",
                    "if Path(\"/kaggle/input\").exists():\n",
                    "    candidates += list(Path(\"/kaggle/input\").rglob(\"run_llama_clean_k2_random_control.py\"))\n",
                    "if not candidates:\n",
                    "    candidates = list(Path(\".\").rglob(\"run_llama_clean_k2_random_control.py\"))\n",
                    "\n",
                    "assert candidates, \"Could not find run_llama_clean_k2_random_control.py\"\n",
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
                    "# Run Matched Random-K2 Control (FP16 + INT4 NCALIB=128)\n",
                    "import os\n",
                    "import subprocess\n",
                    "\n",
                    "cmd = [\n",
                    "    sys.executable, \"-u\", str(SCRIPT_PATH),\n",
                    "    \"--fp16-model\", \"meta-llama/Llama-3.1-8B\",\n",
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
                    "# Display Table 8 Results and inspect generated output files\n",
                    "import pandas as pd\n",
                    "from IPython.display import display, Markdown\n",
                    "\n",
                    "md_files = list(Path(\"/kaggle/working\").rglob(\"llama_clean_k2_random_control_n100_ncalib128.md\"))\n",
                    "csv_files = list(Path(\"/kaggle/working\").rglob(\"llama_clean_k2_random_control_n100_ncalib128.csv\"))\n",
                    "\n",
                    "if md_files:\n",
                    "    content = md_files[0].read_text(encoding=\"utf-8\")\n",
                    "    display(Markdown(\"### Table 8: Llama Clean-K2 Headline vs Matched Random-K2 Control\"))\n",
                    "    display(Markdown(content))\n",
                    "\n",
                    "if csv_files:\n",
                    "    df = pd.read_csv(csv_files[0])\n",
                    "    display(df)\n",
                    "\n",
                    "print(\"Done! You can download llama_clean_k2_random_control_n100_ncalib128.json, .md, and .csv from /kaggle/working.\")\n"
                ],
            },
        ],
    }
    dst.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Created notebook: {dst}")


def write_readme(dest: Path) -> None:
    text = """# Llama Clean-K2 Matched Random-K2 Control (FP16 & INT4 NCALIB=128)

This package implements and runs the matched random-head control ($K=2$) for the Llama clean-K2 headline finding on both FP16 (`meta-llama/Llama-3.1-8B`) and canonical INT4 (`llama31_8b_base_gptq_int4_ncalib128`).

## 1. Protocol Specifications
- **Head selection**: $K=2$ attention heads selected using `random_head_control` with seed 42.
  - Excluded from draw pool:
    - Parametric clean K2 headline heads: `L20_H14`, `L26_H15`.
    - Context K10 heads: `L4_H9`, `L9_H0`, `L1_H11`, `L4_H5`, `L19_H27`, `L17_H12`, `L7_H26`, `L14_H23`, `L3_H22`, `L31_H20`.
  - Drawn heads: **`L20_H24`** and **`L3_H19`**.
- **JuICE steering**:
  - Decode-time dual-run steering ($\alpha_+ = 1.0, \alpha_- = 0.0$), greedy decoding, `max_new_tokens=16`.
- **Evaluation set**:
  - Seed-42 $n=100$ items from `data/conflictqa_pilot250_popular.jsonl` matching Llama Stage 2 diagnostic split.
- **Scoring**:
  - Question-aware matcher (`label_conflict_response` with `question=gold_question`).
  - Inclusive $\\Delta$: Every baseline-context item that leaves context under steering.
  - Gold-PAR-only $\\Delta$: Leave-context counts only if steered completion names the gold parametric entity (`_mentions_answer`).
  - Paired bootstrap: 1000 resamples, seed 42, percentile 95% CI.
- **Output naming**:
  - `llama_clean_k2_random_control_n100_ncalib128.json`
  - `llama_clean_k2_random_control_n100_ncalib128.md`
  - `llama_clean_k2_random_control_n100_ncalib128.csv`

## 2. Running on Kaggle (T4 x 2)
1. Upload `kaggle_llama_clean_k2_random_control.zip` as a dataset.
2. Attach `llama31_8b_base_gptq_int4_ncalib128` as a dataset (or provide checkpoint in working directory).
3. Open `kaggle_llama_clean_k2_random_control.ipynb`.
4. Enable **GPU T4 x2**, enable **Internet**, add Hugging Face token secret (`HF_TOKEN`) if gating requires it.
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
    if not data_src.is_file():
        data_src = ROOT / "kaggle_upload_stage2_n100_headline" / "data" / "conflictqa_pilot250_popular.jsonl"
    shutil.copy2(data_src, out_data / "conflictqa_pilot250_popular.jsonl")

    # 3. Copy directional_heads and build random head JSON
    out_heads = OUT_DIR / "directional_heads"
    out_heads.mkdir(parents=True)
    
    split_src = ROOT / "kaggle_upload_llama_n100_diagnostics" / "directional_heads" / "split.json"
    if not split_src.is_file():
        split_src = ROOT / "kaggle_upload_stage2_n100_headline" / "directional_heads" / "split.json"
    assert split_src.is_file(), f"split.json not found: {split_src}"
    shutil.copy2(split_src, out_heads / "split.json")

    # Copy explicitly from canonical Llama directory
    par_src = LLAMA_CANONICAL_HEADS_DIR / "heads_dir_parametric_k2_clean.json"
    ctx_src = LLAMA_CANONICAL_HEADS_DIR / "heads_dir_context_k10.json"
    assert par_src.is_file(), f"Source clean K2 file not found: {par_src}"
    assert ctx_src.is_file(), f"Source context K10 file not found: {ctx_src}"

    shutil.copy2(par_src, out_heads / "heads_dir_parametric_k2_clean.json")
    shutil.copy2(ctx_src, out_heads / "heads_dir_context_k10.json")

    # Post-copy validation assertion
    copied_par_data = json.loads((out_heads / "heads_dir_parametric_k2_clean.json").read_text(encoding="utf-8"))
    copied_par_keys = [f"L{h['layer']}_H{h['head']}" for h in copied_par_data.get("enlarge", [])]
    assert copied_par_keys == ["L20_H14", "L26_H15"], (
        f"Post-copy validation failed: Expected Llama clean K2 ['L20_H14', 'L26_H15'], got {copied_par_keys}"
    )

    build_random_k2_head_json(out_heads)

    # Also make sure root directional_heads has heads_random_k2.json
    root_heads = ROOT / "directional_heads"
    root_heads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_heads / "heads_random_k2.json", root_heads / "heads_random_k2.json")

    # 4. Copy scripts
    out_scripts = OUT_DIR / "scripts"
    out_scripts.mkdir(parents=True)
    runner_dest = out_scripts / "run_llama_clean_k2_random_control.py"
    shutil.copy2(ROOT / "scripts" / "run_llama_clean_k2_random_control.py", runner_dest)

    # Validate runner script AST to prevent signature regressions (e.g. duplicate precision arg)
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
    shutil.copy2(NOTEBOOK_PATH, OUT_DIR / "kaggle_llama_clean_k2_random_control.ipynb")
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
        assert zip_keys == ["L20_H14", "L26_H15"], (
            f"Zip verification failed: heads in zip are {zip_keys}, expected ['L20_H14', 'L26_H15']"
        )
    print(f"[PACKAGE VERIFIED] Zip archive contains valid Llama clean K2 heads: {zip_keys}")


if __name__ == "__main__":
    main()
