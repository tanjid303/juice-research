# Kaggle — N_CALIB=128 re-quant + 3 Stage 2 cells

Re-quantize Llama-3.1-8B **base** GPTQ-INT4 with a proper calibration set (128, not the 64-sample smoke), then re-run the three locked transfer cells against the **new** checkpoint. Diff vs bundled N=64 results.

**Do not overwrite** `/kaggle/working/llama31_8b_base_gptq_int4` (N_CALIB=64). New weights go to `llama31_8b_base_gptq_int4_ncalib128/`.

Nothing except `N_CALIB` and that output path is changed vs the original quant recipe. Same 50 eval ids, same FP16 heads, same random-K seed/exclusion, same bootstrap (1000 draws, seed 42). JuICE dual-run port is copied unchanged.

---

## Run order (one session)

Upload this zip as a Kaggle dataset. **File → Upload notebook** `juice_repro/kaggle_stage2_int4_ncalib128.ipynb`. Save & Run All.

| Order | Script | Does | Expected runtime |
|------:|--------|------|------------------|
| 1 | `scripts/ncalib128_01_quantize.py` | GPTQ 4-bit gs128, C4×128 truncated to 512 tok, save `llama31_8b_base_gptq_int4_ncalib128/` | **45–90 min** |
| 2 | `scripts/ncalib128_02_rerun_three_cells.py` | baseline + context K10 + random-K10 + parametric clean K2; write `int4_transfer_bootstrap.json` | **3–6 h** |
| 3 | `scripts/ncalib128_03_compare.py` | table N64 vs N128 | **<1 min** |
| 4 | `scripts/ncalib128_04_native_clean_k2.py` | steer native INT4 clean pair `L16_H30` + `L17_H27` (K=2, α+=1); writes `int4_native_clean_k2_summary.json` | **1.5–3 h** |

**Total ~5.5–11 h** on 2× T4 (12 h limit). Each stage prints `[elapsed]`.

Stage 4 is independent of stage 3 and only needs the stage-1 checkpoint. If stages 1–2 ran long (past ~9 h), skip stage 4 and run it in a second session with this zip plus the stage-1 Save Version attached.

### Notebook settings

| Setting | Value |
|---------|--------|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** |
| Secret | `HF_TOKEN` |
| Datasets | this zip only (script 01 builds the checkpoint) |

Accept the Llama-3.1-8B license on Hugging Face.

If you already have the N128 checkpoint from a previous Save Version, attach that notebook output and script 01 will skip (folder already has `config.json` + weights). Script 02 still **refuses** the old `llama31_8b_base_gptq_int4` (N=64) path.

---

## What is locked (do not resample)

- Eval: 50 ids in `directional_heads/split.json` (`seed` 42)
- Context K10 heads: `directional_heads/heads_dir_context_k10.json` (FP16 directional top-10)
- Parametric clean K2: `L20_H14`, `L26_H15` in `heads_dir_parametric_k2_clean.json`
- Native INT4 clean K2 (stage 4): `L16_H30`, `L17_H27` in `heads_int4_native_parametric_k2_clean.json` — identified on INT4, Jaccard 0.00 vs the FP16 pair, so **no FP16 same-sign gate applies**
- Random-K10: `random_head_control(n_layers, n_heads, 10, seed=42, exclude=FP16 context K10)` — same as `juice_repro/_make_int4_native_id_nb.py` cell 7
- Prompt: `juice_coherent`, α+=1, α−=0, `max_new_tokens=16`, greedy
- Bootstrap: `common.stats_analysis.bootstrap_paired_follow_delta`, `n_boot=1000`, `seed=42`
- Quant: gptqmodel 2.2 + transformers 4.51.3, bits=4, group_size=128, sym=True, desc_act=True, damp=0.01, C4 `en/c4-train.00001-of-01024.json.gz`, MAX_SEQ_LEN=512, BACKEND.TORCH (T4)

## Outputs

```
/kaggle/working/llama31_8b_base_gptq_int4_ncalib128/   # new weights + quant_meta.json
/kaggle/working/juice_stage2_int4_ncalib128/
    steer_baseline.json
    steer_context_k10.json
    random_context_k10.json
    steer_parametric_k2_clean.json
    int4_transfer_summary.json
    int4_transfer_bootstrap.json    # same schema as int_4_result/int4_transfer_bootstrap.json
    int4_random_k_summary.json      # same schema as here_is_the/int4_random_k_summary.json
    n64_vs_n128_comparison.json
    steer_int4_native_parametric_k2_clean.json   # stage 4 per-item rows
    int4_native_clean_k2_summary.json            # stage 4, transfer-cell schema
    int4_native_clean_k2_bootstrap.json          # stage 4, bootstrap schema
```

Comparison table columns: `condition | N64 Δ [CI] | N128 Δ [CI] | same sign? | excludes 0 verdict flip?`

Bundled N=64 baselines (for script 03, not re-used as weights): `n64_baseline/`.

Bundled completed N=128 stage-2 outputs: `ncalib128_reference/`. Stage 4 compares its
in-session baseline against `ncalib128_reference/steer_baseline.json` item-by-item and
records the agreement in `baseline_vs_ncalib128_reference`, so the new Δ stays on the
same footing as the other N=128 transfer cells even in a fresh session.
