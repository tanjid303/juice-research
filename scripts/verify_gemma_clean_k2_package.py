import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "kaggle_gemma_clean_k2_int4_transfer.zip"
NB_PATH = ROOT / "kaggle_gemma_clean_k2_int4_transfer.ipynb"

assert ZIP_PATH.is_file(), f"Zip file not found: {ZIP_PATH}"
assert NB_PATH.is_file(), f"Notebook file not found: {NB_PATH}"

with zipfile.ZipFile(ZIP_PATH) as zf:
    # 1. Clean K2 heads
    par_data = json.loads(zf.read("directional_heads/heads_dir_parametric_k2_clean.json").decode("utf-8"))
    par_keys = [f"L{h['layer']}_H{h['head']}" for h in par_data["enlarge"]]
    assert par_keys == ["L31_H7", "L39_H6"], f"Clean K2 heads mismatch: {par_keys}"
    print(f"[OK] Clean K2 heads in zip: {par_keys}")

    # 2. Random K2 heads
    rand_data = json.loads(zf.read("directional_heads/heads_random_k2.json").decode("utf-8"))
    rand_keys = [f"L{h['layer']}_H{h['head']}" for h in rand_data["enlarge"]]
    assert rand_keys == ["L41_H11", "L7_H3"], f"Random K2 heads mismatch: {rand_keys}"
    print(f"[OK] Random K2 heads in zip: {rand_keys}")

    # 3. Context K10 & raw peak
    ctx_data = json.loads(zf.read("directional_heads/heads_dir_context_k10.json").decode("utf-8"))
    assert len(ctx_data["enlarge"]) == 10, "Context K10 length mismatch"
    raw_data = json.loads(zf.read("directional_heads/heads_dir_parametric_k1_raw.json").decode("utf-8"))
    assert [f"L{h['layer']}_H{h['head']}" for h in raw_data["enlarge"]] == ["L27_H6"], "Raw peak mismatch"
    print("[OK] Context K10 (10 heads) and Raw peak (L27_H6) in zip.")

    # 4. Split
    split_data = json.loads(zf.read("directional_heads/split.json").decode("utf-8"))
    assert split_data["n_eval"] == 100, f"Split n_eval mismatch: {split_data['n_eval']}"
    assert len(split_data["eval_ids"]) == 100, "Split eval_ids count mismatch"
    assert split_data["seed"] == 42, "Split seed mismatch"
    print("[OK] Preregistered split: seed 42, n_eval 100.")

    # 5. Check model IDs and absence of Llama checkpoints
    runner_text = zf.read("scripts/run_gemma_clean_k2_int4_transfer.py").decode("utf-8")
    nb_text = zf.read("kaggle_gemma_clean_k2_int4_transfer.ipynb").decode("utf-8")

    assert "llama31_8b_base_gptq_int4_ncalib128" not in runner_text, "Found Llama int4 in runner!"
    assert "llama31_8b_base_gptq_int4_ncalib128" not in nb_text, "Found Llama int4 in notebook!"
    assert "meta-llama/Llama-3.1-8B" not in runner_text, "Found Llama fp16 in runner!"
    assert "meta-llama/Llama-3.1-8B" not in nb_text, "Found Llama fp16 in notebook!"

    assert "gemma2_9b_base_gptq_int4_ncalib128" in runner_text, "Missing Gemma int4 in runner!"
    assert "gemma2_9b_base_gptq_int4_ncalib128" in nb_text, "Missing Gemma int4 in notebook!"
    assert "google/gemma-2-9b" in runner_text, "Missing Gemma fp16 in runner!"
    assert "google/gemma-2-9b" in nb_text, "Missing Gemma fp16 in notebook!"
    print("[OK] Model names verified: 'google/gemma-2-9b' and 'gemma2_9b_base_gptq_int4_ncalib128'.")

    # 6. Notebook JSON integrity & HF auth cell
    nb_json = json.loads(nb_text)
    assert nb_json["cells"], "Notebook has no cells!"
    auth_cell = any("kaggle_secrets" in "".join(c.get("source", [])) for c in nb_json["cells"])
    assert auth_cell, "Notebook missing Kaggle Secrets HF auth cell!"
    print("[OK] Kaggle notebook JSON valid and includes HF auth cell.")

print("\nALL PACKAGE VERIFICATION TESTS PASSED SUCCESSFULLY!")
