# Progress writeup

**Project:** Do JuICE causal head-steering interventions transfer from FP16 to GPTQ-INT4 on Llama-3.1-8B?

**Status as of 29 Aug 2026:** Stage 1 complete. Stage 2 FP16 validated on base. FP16→INT4 transfer: only **context K10** excludes zero. Native INT4 ID + random-K done. **Parametric recovery Kaggle done:** L1_H4 at α=1 is **wrong-sign** (+6, CI includes 0). Clean-pair α-grid: α=1 still −2 (null); α=3/5 −12 excludes 0 but is **`other` inflation** (25–32/50), not parametric. Suppress ctx K10 = −2 (null). Parametric transfer is not rescued by a different head or by rescaling.

**Full design:** `PROJECT_BRIEF_FULL.md`  
**Stage 1 decision memo:** `writeup/mini_pilot.md`

---

## 1. Where we are in the pipeline

| Stage | Goal | Status |
|-------|------|--------|
| **Stage 1** — Behavioral MR★ / AME under conflict | Does INT4 shift source arbitration at 8B? | **Done — go** |
| **Stage 2 FP16** — JuICE port + head ID + steering gate | Does dual-run JuICE steer our target model before quant? | **Done on base**; Instruct parametric arm weak |
| **Stage 2 INT4** — 2×2 identify×apply + controls | Does FP16 steering transfer to INT4? | **Done** (transfer + native ID + Jaccard + random-K + L1_H4/α-grid/suppress). Context K10 transfers; parametric does not. |
| **Stage 3** — Repair (re-ID / rescale) | Fix transfer if it breaks | **Open.** Rescale of the directional pair is not a clean fix (α≥3 dumps to `other`). L1_H4 is not a substitute causal head. |
| **Fix 1** — A/B/C separability on cheap model | De-risk repair protocol | Not started |

---

## 2. Stage 1 — Behavioral mini-pilot (completed)

Full numbers and diagnostics: **`writeup/mini_pilot.md`**.

### 2.1 Llama-3.2-1B-Instruct — pivot (b)

- **MR★ Δ = +1.7 pp**, bootstrap CI includes 0 → **no reliable arbitration shift**.
- Apparent context-following drop was a **brevity / under-commitment confound** (INT4 outputs shorter, more `other`).
- **Decision:** do not run JuICE on 1B items; use 1B only as a null screen.

### 2.2 Llama-3.1-8B-Instruct — go to Stage 2

- **Data:** `data/conflictqa_pilot250_popular.jsonl` (250 items)
- **Models:** FP16 `meta-llama/Llama-3.1-8B-Instruct` vs GPTQ-INT4 `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`
- **Hardware:** Kaggle 2× Tesla T4; INT4 via **gptqmodel** `BACKEND.TORCH` (Marlin JIT failed on sm_75)
- **Double-filter n:** 164 (closed-book correct at FP16 **and** INT4)

| Metric | FP16 | INT4 | Δ |
|--------|------|------|---|
| MR★ (shared 164) | 46.3% | 22.6% | **−23.8 pp** |
| Bootstrap 95% CI | — | — | **[−31.7, −17.1]** (excludes 0) |
| `follows_context` AME (corrected scorer) | — | — | **~+10 pp** (cluster-robust / raw) |

**Label shift on shared 164:** parametric 76→37 (−39); context 52→69 (+17); rejects_context 13→22; other 23→36.

**Scorer fix:** negation/denial frames → `rejects_context` (not false `context`); bare entity names → `context`. Pre-fix AME (+20.6 pp) was inflated; publish **~+10 pp**.

**Decision:** §5.1 gate passed. Proceed to Stage 2 JuICE on **8B** (not 1B).

**Open QC note:** third-party INT4 checkpoint showed variable PPL (+2.5% to +27.4% across samples). Coherence + structural QC were sufficient to proceed; self-quantize before final paper numbers.

---

## 3. Stage 2 — JuICE FP16 port and validation

### 3.1 Implementation decision

Upstream JuICE (`external/JUICE`) targets Gemma / nnsight and raises on Llama in `Replace_AttnModule`. We **ported the dual-run mechanism** rather than importing upstream directly.

**Ported mechanism (matches JuICE paper/code intent):**

1. Post-`o_proj` head contribution: `attn_slice @ W_o_slice`
2. **Pass 1:** record per-head outputs
3. **Pass 2:** `o_proj.output += alpha * saved`
4. **Generation:** two full forwards per decode token (not prefill-only cache reuse)

**Key files:**

| File | Role |
|------|------|
| `common/juice_attn.py` | `GetHeadsLlamaAttention` wrapper |
| `common/juice.py` | ID, steering, directional rerank, eval prompts |
| `common/prompts.py` | JuICE 3-type prompts (`clean`, `substitution_conflict`, `coherent_conflict`) |
| `common/models.py` | `FP16_8B_MODEL_ID`, `FP16_8B_BASE_MODEL_ID` |
| `stage2_mechanistic/` | CLI scripts: `identify_heads.py`, `apply_steering.py`, etc. |

**Important:** An earlier approximate implementation (pre-o_proj slices, o_proj input injection, prefill cache reused in decode) was **wrong** and was replaced.

**K=10** enlarge heads for Llama-class models (JuICE convention for 7–8B).

### 3.2 Eval protocol (locked for Cell-3 runs)

- **Held-out set:** 50 items from `split.json` (seed=42; 25 calib / 50 eval IDs)
- **Metric:** `follows_context` = conflict label `== "context"` (excludes `rejects_context`)
- **Coherence check:** reject `://` spam, all-punct, repeated-token garbage; require minimum alphabetic content
- **Steering default for Cell-3:** α+=1, α−=0 (enlarge-only)

---

## 4. Stage 2 runs — chronological

### 4.1 Instruct FP16 initial reproduction (`juice_repro/kaggle_stage2_fp16_8b.ipynb`)

**Model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Prompts:** Instruct chat template (`context_conflicts`)  
**Artifacts:** user download `juice_stage2_fp16/`

| Condition | follows_context | Notes |
|-----------|-----------------|-------|
| Baseline | **28%** | — |
| Steer context (raw ID heads) | ~28% | weak |
| Steer parametric | ~28% | weak |
| α=5 steering | **collapsed** | all `other`, garbage `://://` |

**Verdict:** Context/parametric steering at default α did not clearly beat baseline; high α destroys coherence.

**Head ID diagnostics (Instruct, n_calib=25, chat prompts):**

| Metric | Value |
|--------|-------|
| Raw enlarge Jaccard (ctx vs par sets) | **0.54** |
| Peak ΔP (ctx / par) | ~0.008 |
| Directional top-10 Jaccard | **0.0** |
| Best directional_par score | ~**2.7×10⁻⁴** (tiny) |

---

### 4.2 Instruct directional alpha sweep (`juice_repro/kaggle_stage2_cell3_alpha_sweep.ipynb`)

Offline directional re-rank on Instruct ID scores; no re-ID.

**Artifacts:** `juice_stage2_alpha_sweep/`, `direction-output/`

| Condition | fc | Δ vs baseline | Coherence |
|-----------|-----|---------------|-----------|
| Baseline | 28% | — | 100% |
| Context K=10, α+=1 | 38% | **+10 pp** | 100% |
| Context K=5, α+=1 | — | — | — |
| Parametric, α+=1 | 32% | +4 pp | 100% |
| α+=2, α−=1 | degraded | negative | **collapse** (~48–76%) |

At α+=1 both directions moved in the sweep aggregate; high α is unsafe.

**α+=2, α−=1 (same summary JSON, second row):** context fc 16% (−12 pp), coherence **76%**; parametric fc 6% (−22 pp), coherence **48%**; labels collapse into `other` (38 / 47 of 50). This is why INT4 transfer locked α+=1, α−=0.

---

### 4.3 Instruct directional Cell-3 (`juice_repro/kaggle_stage2_cell3_directional.ipynb`)

Same 50 eval items; directional heads; α+=1, α−=0.

**Artifacts:** `direction-output/directional_cell3_summary.json`

| Condition | Heads | fc | Δ pp | Coherence | P/C/O |
|-----------|-------|-----|------|-----------|-------|
| Baseline | — | 28% | — | 100% | 24/14/12 |
| steer_context_k10 | directional ctx top-10 | 36% | **+8** | 100% | 11/18/21 |
| steer_context_k5 | directional ctx top-5 | 34% | **+6** | 100% | 16/17/17 |
| steer_parametric_k2 | L2_H23, L8_H9 | 28% | **0** | 100% | 26/14/10 |
| steer_parametric_k3 | + L0_H2 | 26% | **−2** | 100% | 24/13/13 |

**Verdict:** Context steering works at low α on Instruct. **Parametric steering does not move fc meaningfully** (0/−2 pp) despite directional re-rank.

---

### 4.4 Base model ID diagnostic (`juice_repro/kaggle_stage2_base_id_diagnostic.ipynb`)

**Model:** `meta-llama/Llama-3.1-8B` (base)  
**Calibration:** |D|=4, JuICE 3-type **plain** prompts (no chat template)  
**Artifacts:** `juice_stage2_base_id/base_id_comparison_summary.json`

| Metric | Instruct | Base |
|--------|----------|------|
| Raw enlarge Jaccard | 0.54 | **0.053** |
| Peak raw ΔP | ~0.008 | **0.036 / 0.058** |
| Best directional_par | 2.7×10⁻⁴ | **0.304** (~1100×) |
| Directional top-10 Jaccard | 0 | 0 |

**Interpretation:** Instruct/chat **suppresses parametric-directional ID signal**. Dual-run port is likely OK; the failure mode is model + prompt format, not a broken hook.

**Head caveats (parametric top ranks):**

- **L20_H14, L26_H15** — cleaner (positive `par_pos`, strong directional_par)
- **L31_H14** — context-suppression-heavy (both ctx and par probs down under ID); noisier when included in steering

**Directional head JSONs built:** `juice_stage2_base_id/directional_heads/heads_dir_*.json`

---

### 4.5 Base directional Cell-3 (`juice_repro/kaggle_stage2_base_cell3_directional.ipynb`)

**Model:** `meta-llama/Llama-3.1-8B` base  
**Prompts:** JuICE `coherent_conflict` plain completion (`prompt_format="juice_coherent"`)  
**Artifacts:** `juice_stage2_base_directional_cell3/base_directional_cell3_summary.json`

| Condition | fc | Δ pp | Coherence | P/C/O |
|-----------|-----|------|-----------|-------|
| Baseline | **16%** | — | 100% | 40/8/2 |
| steer_context_k10 | 26% | **+10** | 100% | 34/13/3 |
| steer_context_k5 | 22% | **+6** | 100% | 36/11/3 |
| steer_parametric_k2 (L20_H14, L31_H14) | 10% | **−6** | 100% | 39/5/6 |
| steer_parametric_k3 (+ L26_H15) | 10% | **−6** | 100% | 39/5/6 |
| steer_parametric_k2_clean (L20_H14, L26_H15) | 8% | **−8** | 100% | 42/4/4 |

**Flip analysis (vs baseline):**

- Context K10: **+5** net par→ctx, 0 ctx→par
- Parametric K2: **−3** net context labels (5 ctx→par, 2 par→ctx)
- Clean K2: **−4** net (4 ctx→par, **0** par→ctx regressions) — strongest, most one-directional

**Example context win:** `7835` La Paz → Sucre; `7843` long parametric blurb → `Burlington`  
**Example parametric push:** fewer context-following answers; clean heads avoid back-sliding to context vs full K2 with L31_H14

**Stage 2 FP16 gate: PASSED on base model.**

---

## 5. Cross-run comparison (what we learned)

### 5.1 Instruct vs base (Fix 3 confound check)

| Finding | Instruct | Base |
|---------|----------|------|
| Context steering at α=1 | **Works** (+6–8 pp fc) | **Works** (+6–10 pp fc) |
| Parametric steering at α=1 | **Fails** (0/−2 pp) | **Works** (−6 to −8 pp) |
| ID parametric signal | ~10⁻⁴ | ~0.30 |
| Baseline fc (not comparable) | 28% (chat) | 16% (plain JuICE) |

**Conclusion:** JuICE dual-run implementation is sound. Instruct + chat template is a poor mechanistic target for parametric steering; **use base model + base-ID heads** for INT4 transfer experiments.

### 5.2 What we have *not* shown yet

- JuICE does **not** yet match Li et al. table cells numerically on overlapping settings (direction + gap vs baseline is enough for our gate, but paper may want closer reproduction on Gemma/ParaConflict).
- **FP16→INT4 transfer + native ID + Jaccard + random-K + parametric recovery (L1_H4 / α-grid / suppress) are done.** Reverse transfer still open. Native-apply of INT4 directional top-2 was already in the transfer cell (0 pp).
- **Fix 1** A/B/C pilot on cheap model not run.

---

## 6. Decisions locked by this work

1. **Proceed to INT4 on Llama-3.1-8B base**, not Instruct, for mechanistic transfer.
2. **Canonical parametric heads:** clean K=2 (**L20_H14, L26_H15**); context K=5 and K=10 from base directional rerank.
3. **Steering gain:** α+=1, α−=0 for first INT4 cells (α=2/α−=1 collapses coherence on Instruct).
4. **Eval split:** keep `split.json` 50-item held-out set for comparability across FP16/INT4.
5. **Prompt format:** base gets JuICE plain `coherent_conflict`; do not mix chat template on base runs.
6. **Do not** use Instruct-identified parametric heads for INT4 gate — they do not steer behavior.

---

## 7. Stage 2 INT4 — FP16→INT4 transfer (done)

Per `PROJECT_BRIEF_FULL.md` §5.5:

| Cell | Identify on | Apply to | Status |
|------|-------------|----------|--------|
| Reproduction | FP16 | FP16 | Done (base) |
| **Transfer** | FP16 | **INT4** | **Done — gate passed** |
| Native | INT4 | INT4 | **ID + Jaccard + random-K done** (native-apply of INT4 heads not run) |
| Reverse | INT4 | FP16 | Lower priority |

**Run:** self-quantized `llama31_8b_base_gptq_int4` (gptqmodel 2.2, 4-bit gs128, `desc_act=True`, N_CALIB=64 smoke). Same 50 eval items, `juice_coherent`, α+=1, α−=0. FP16 heads reused (no re-ID). Artifacts: `int_4_result/`.

**Gate (pre-registered):** primary conditions context K10/K5 + parametric clean K=2 keep the **same sign** as FP16 and coherence ≥ 0.90. Magnitude match was not required.

Paired bootstrap: resample the **50 eval items with replacement, 1000 draws**, Δ = steered `follows_context` − baseline (paired on item id). 95% percentile CI. Grid is 2 pp (1/50).

| Condition | Heads | INT4 fc | INT4 Δ | 95% CI | excl. 0? | FP16 Δ | FP16 95% CI | Sign | Coh | P/C/O |
|-----------|-------|---------|--------|--------|----------|--------|-------------|------|-----|-------|
| baseline | none | 14% | — | — | — | 16% | — | — | 100% | 39/7/4 |
| context K10 | directional ctx top-10 | 26% | **+12** | **[+4, +22]** | **yes** | +10 | [+2, +20] | OK | 100% | 34/13/3 |
| context K5 | directional ctx top-5 | 20% | +6 | [−2, +16] | no | +6 | [0, +12] | OK | 100% | 37/10/3 |
| parametric clean K2 | L20_H14, L26_H15 | 12% | **−2** | **[−6, 0]** | **no** | −8 | [−16, −2] | OK | 100% | 41/6/3 |
| parametric K2 | L20_H14, L31_H14 | 14% | 0 | [−10, +10] | no | −6 | [−16, +4] | FLIP | 100% | 37/7/6 |
| parametric K3 | + L26_H15 | 10% | −4 | [−12, +6] | no | −6 | [−18, +6] | OK | 100% | 35/5/10 |

**Statistical read (this is the correction):**

- **Context K10 transfers and is distinguishable from zero.** INT4 +12 pp, CI [+4, +22]; 997/1000 bootstrap draws positive. Same as FP16 (CI [+2, +20] also excludes 0).
- **Context K5 matches FP16 (+6) but both CIs include 0.** INT4 [−2, +16]; FP16 [0, +12]. Point estimate is stable across precisions; n=50 cannot separate it from noise.
- **Canonical parametric −2 pp does not survive two-sided scrutiny.** CI [−6, 0] includes zero. 622/1000 draws negative, **378 zero, 0 positive** — the sign is consistently non-positive, but the magnitude is one extra context-following item vs baseline and is compatible with a true delta of 0. FP16 clean K2 **does** exclude zero (−8 pp, [−16, −2]). The “weaker not collapsed” claim is therefore: INT4 parametric is **not a sign flip and not a collapse**, but it is **not a detected effect**. Treating −2 pp as a real attenuation of an 8 pp object is not justified on this sample.
- Parametric K3 −4 pp CI [−12, +6] is even noisier (136/1000 draws positive).
- Noisy parametric K2 (L31_H14) remains 0 pp with a CI that straddles zero symmetrically.
- **Coherence 100%** on every condition — attenuation is not a garbage-output artifact.
- INT4 baseline 14% vs FP16 16% is a one-item difference.

**Implication:** The pre-registered **same-sign gate still passed**. After CIs, the mechanistic claim that survives is **context K10**. Parametric transfer is unproven at α=1. Native INT4 ID / random-K / Jaccard (§7.1) do not rescue the directional clean pair. L1_H4 and the α-grid (§7.3) close the remaining recovery hypotheses: the raw-ID peak steers the **wrong way**; turning α up replaces answers with `other`, not parametric.

---

## 7.1 Native INT4 ID + Jaccard + random-K (done)

**Artifacts:** `juice_stage2_int4_native_id/` plus Kaggle dump `here_is_the/` (`int4_random_k_summary.json`, per-condition `random_*.json`). Directional rerank + Jaccard from saved per-item maps.

Same protocol as FP16 base: |D|=4, 3-prompt-type JuICE prompts, FAST alphas. Peak ΔP similar to FP16 (ctx 0.045 / par 0.058 vs FP16 0.036 / 0.058). Raw enlarge Jaccard ctx-vs-par on INT4 = **0.0**.

**INT4 raw enlarge K10 (from `int4_native_id_summary.json`):**

- context: L4_H5, L5_H1, L14_H23, L31_H20, L14_H31, L16_H28, L14_H20, L8_H31, L12_H9, L2_H31
- parametric: **L1_H4**, L2_H18, L10_H18, L11_H2, L19_H20, L16_H3, L24_H20, L1_H23, L3_H1, L17_H6

**INT4 directional sets:** ctx K10 = L7_H26, L14_H10, L7_H21, L4_H5, L3_H3, L31_H20, L4_H11, L14_H23, L8_H29, L9_H0; par K2 = L20_H14, L31_H14; par K3 = + L8_H9; “clean” K2 = L16_H30, L17_H27.

### Do the FP16 heads come back?

**Parametric directional — mostly yes, but not the “clean” pair.**

| Head | FP16 role | INT4 directional_par rank | INT4 par_pos | INT4 ctx_pos |
|------|----------|----------------------------|--------------|--------------|
| L20_H14 | clean K2 + noisy K2/K3 | **#1** | **−0.044** (was +0.007 on FP16) | −0.301 |
| L31_H14 | noisy K2/K3 (ctx-suppression) | **#2** | −0.104 | −0.360 |
| L26_H15 | clean K2 + K3 | **#4** | **−0.002** (was +0.018 on FP16) | −0.185 |

INT4 directional top-2 = **L20_H14, L31_H14** — **exact match** to FP16 noisy K2 (the set that transferred at **0 pp**). L26_H15 is still in the top handful. The FP16 clean filter (`par_pos>0` and `dir_par>0`) on INT4 yields **L16_H30, L17_H27**, not L20_H14/L26_H15 — because both canonical heads now have **negative** `par_pos`. On INT4 they look like L31_H14: strong context suppression, not a parametric-token boost.

**Context directional — partial overlap, noisy ranking.**

| Set | INT4 | FP16 | overlap | Jaccard |
|-----|------|------|---------|---------|
| directional K10 | L7_H26, L14_H10, L7_H21, L4_H5, L3_H3, L31_H20, L4_H11, L14_H23, L8_H29, L9_H0 | L4_H9, L9_H0, L1_H11, L4_H5, L19_H27, L17_H12, L7_H26, L14_H23, L3_H22, L31_H20 | 5/10 | **0.33** |
| directional K5 | L7_H26, L14_H10, L7_H21, L4_H5, L3_H3 | L4_H9, L9_H0, L1_H11, L4_H5, L19_H27 | L4_H5 only | **0.11** |
| raw enlarge K10 ctx | … | … | L14_H23, L14_H31, L31_H20 | 0.18 |
| raw enlarge K10 par | … | … | L1_H4, L10_H18, L24_H20 | 0.18 |

FP16 K10 heads on INT4 directional_ctx ranks: L7_H26 **#1**, L4_H5 #4, L31_H20 #6, L14_H23 #8, L9_H0 #10; L4_H9 is #11 (just off the transferred set); L19_H27 is **#83**.

### Jaccard vs |D|=4 bootstrap noise floor (25 draws, per-item score maps)

| Spec | Observed J (FP16 vs INT4) | INT4-vs-INT4 noise mean [95%] | vs noise |
|------|----------------------------|-------------------------------|----------|
| dir context K10 | 0.33 | 0.27 [0.05, 0.82] | **within** — K10 ranking is jittery at \|D\|=4 |
| dir context K5 | 0.11 | 0.19 [0.00, 0.43] | **within** |
| dir parametric K2 | **1.00** | 0.95 [0.33, 1.00] | **within** — but noise floor is already ~1; top-2 is **stable** |
| dir parametric K3 | 0.50 | 0.74 [0.50, 1.00] | **within** (at the floor edge) |
| dir parametric clean K2 | **0.00** | 0.24 [0.00, 1.00] | **within** — clean filter is too unstable to interpret |

### Read with transfer CIs + random-K

Same 50 eval items, α+=1, α−=0, `juice_coherent`. Random heads exclude the FP16 identified set. New-session baseline matched transfer (fc **14%**, 39/7/0/4, coherence 100%). Source: `here_is_the/int4_random_k_summary.json`.

| Condition | Heads (random) | fc | Δ [95% CI] | excl. 0? | Coh | P/C/O | Identified Δ |
|-----------|----------------|----|------------|----------|-----|-------|--------------|
| random context K10 | L20_H23, L3_H19, L0_H25, L24_H0, L8_H30, L7_H31, L7_H8, L4_H18, L23_H27, L3_H9 | 18% | **+4 [−4, +12]** | no | 100% | 38/9/3 | **+12 [+4, +22]** |
| random context K5 | L1_H7, L9_H10, L22_H17, L24_H21, L4_H23 | 16% | +2 [0, +6] | no | 100% | 38/8/4 | +6 [−2, +16] |
| random parametric K2 | L13_H2, L16_H20 | 18% | +4 [0, +10] | no | 100% | 37/9/4 | −2 [−6, 0] |
| random parametric K3 | L8_H22, L13_H11, L15_H19 | 16% | +2 [0, +6] | no | 100% | 39/8/3 | −4 [−12, +6] |

| Condition | Identified transfer Δ [95% CI] | Random-K Δ [95% CI] | Native ID |
|-----------|--------------------------------|---------------------|-----------|
| context K10 | **+12 [+4, +22]** excl. 0 | **+4 [−4, +12]** includes 0 | overlap 5/10, within \|D\|=4 noise |
| context K5 | +6 [−2, +16] includes 0 | +2 [0, +6] includes 0 | J=0.11, within noise |
| parametric clean K2 | −2 [−6, 0] includes 0 | (random K2) **+4 [0, +10]** includes 0 | clean pair **not** recovered |
| parametric K2 L20+L31 | **0 [−10, +10]** | **+4 [0, +10]** | **this is native INT4 top-2** |
| parametric K3 | −4 [−12, +6] includes 0 | +2 [0, +6] includes 0 | top-3 = L20, L31, L8_H9 |

Random-K coherence 100% on every arm — not a collapse control.

**Random-K read:** Context K10’s +12 is **head-specific**, not “any 10 heads at α=1.” Random K10 is +4 and does not exclude zero; identified K10 does. CIs still overlap at n=50, so this is a control-fails / treatment-detects pattern, not a proven difference-of-deltas. Parametric identified sets do not beat random: random K2/K3 move **the wrong way** (more context-following); native INT4’s own top-2 transferred at 0 pp.

### 7.2 Check 2 — directional_par top-20 (local, from native ID scores)

Not a new Kaggle GPU job. Computed from `juice_stage2_int4_native_id/id_diagnostics_raw.json` → `directional_par_top20.json`. Noise proxy (median max(|ctx|,|par|)) ≈ 9.2×10⁻³; 5× threshold ≈ 0.046.

Ranks 1–5 are all **negative** `par_pos` (L20_H14, L31_H14, L8_H9, L26_H15, L14_H16). The INT4 “clean” filter hits L16_H30 / L17_H27 at ranks 6–7 with `par_pos` +0.020 / +0.015 (near noise).

**Only L1_H4** in/near the top-20 has `par_pos` above 5× noise:

| | FP16 | INT4 |
|--|------|------|
| `par_pos` | **+0.0577** (raw enlarge **#1**) | **+0.0578** (raw enlarge **#1**) |
| `ctx_pos` | −0.0392 | −0.0581 |
| `dir_par` | +0.0969 | +0.1159 |
| directional rank | **29** | **14** |
| used in Cell-3 / transfer? | no | no |

Same-head, not INT4-only: GPTQ kept the raw parametric peak. Directional rerank skipped it on both precisions because L20/L31 win on context *suppression*. Steering this head is Test 4 in §7.3 — it **did not** produce a parametric effect.

### 7.3 Parametric recovery Kaggle — **done**

**Artifacts:** `here_is_the-2/int4_parametric_fixes_summary.json` (per-condition `steer_*.json`). Same 50 eval items, `juice_coherent`, baseline **14%** (39/7/4), coherence heuristic 100% on every arm.

| Condition | Heads | α | fc | Δ [95% CI] | excl. 0? | Coh* | P/C/O |
|-----------|-------|---|----|------------|----------|------|-------|
| baseline | — | — | 14% | — | — | 100% | 39/7/4 |
| **L1_H4 K=1** | L1_H4 | +1 / 0 | 20% | **+6 [−2, +14]** | **no** | 100% | 36/10/4 |
| clean K2 | L20_H14, L26_H15 | 0.5 / 0 | 16% | +2 [0, +8] | no | 100% | 39/8/3 |
| clean K2 | same | 1 / 0 | 12% | −2 [−6, 0] | no | 100% | 41/6/3 |
| clean K2 | same | 2 / 0 | 12% | −2 [−12, +8] | no | 100% | 35/6/9 |
| clean K2 | same | 3 / 0 | 2% | **−12 [−22, −2]** | **yes** | 100%* | **24/1/25** |
| clean K2 | same | 5 / 0 | 2% | **−12 [−22, −2]** | **yes** | 100%* | **17/1/32** |
| suppress ctx K10 | FP16 ctx K10 as H− | 0 / 1 | 12% | −2 [−6, 0] | no | 100% | 44/6/0 |

\* `looks_coherent` (no `://` spam / has letters). At α=3/5 this stays 100% while **`other` is 50–64%** — fluent-looking but not a conflict answer.

**Test 4 — L1_H4:** Point estimate is **wrong-sign** for a parametric enlarge (more context-following, not less). 882/1000 bootstrap draws positive, 42 negative, 76 zero. CI includes 0, so it is not a detected context effect either — but it is the opposite of the recovery we needed. The “same raw peak, just steer L1_H4” story is **rejected**.

**Check 1 — alpha grid:** α=1 **replicates** the transfer null (−2, [−6, 0], 0/1000 draws positive). α=0.5 is slightly the wrong way. α=2 is still −2 with a wide CI and `other` starting to rise (9/50). α=3 and α=5 both −12 with CI excluding 0, but parametric labels fall 39→24→17 and `other` rises 4→25→32. Context does not convert to parametric; both conflict labels dump into `other` (rambling / blank-slot / off-entity completions). **Do not read −12 as recovered parametric steering.** Rescale will not restore the FP16 −8 pp object.

**Check 3 — suppress ctx K10:** −2 [−6, 0], same as clean K2 at α=1; 0/1000 draws positive. Parametric count 39→44, context 7→6, `other` 4→0. A one-item nudge, not a recovery path.

**Read:** Parametric transfer is not “dead because we steered the wrong INT4 head.” The only head with real +`par_pos` does not push parametric at α=1. The FP16 clean pair does not become a real parametric effect at any usable α. High α is an `other` collapse that the letter-based coherence check missed. The surviving Stage 2 INT4 claim remains **context K10**.

---

## 8. Artifact index

| Artifact | Path |
|----------|------|
| Stage 1 memo | `writeup/mini_pilot.md` |
| Stage 1 8B Kaggle notebook | `mini_pilot/kaggle_llama31_8b_fp16_vs_int4.ipynb` |
| Data | `data/conflictqa_pilot250_popular.jsonl` |
| Eval split | `juice_stage2_base_id/directional_heads/split.json` |
| Instruct FP16 initial | user download `juice_stage2_fp16/` (not in this repo) |
| Instruct Cell-3 results | `direction-output/directional_cell3_summary.json` |
| Instruct alpha sweep | `juice_stage2_alpha_sweep/alpha_sweep_summary.json` |
| Base ID results | `juice_stage2_base_id/` (`base_id_comparison_summary.json`) |
| Base Cell-3 results | `juice_stage2_base_directional_cell3/base_directional_cell3_summary.json` |
| Base directional heads | `juice_stage2_base_id/directional_heads/heads_dir_*.json` |
| Quantize notebook | `juice_repro/kaggle_quantize_llama31_8b_base_gptq.ipynb` |
| INT4 checkpoint (Kaggle) | notebook output `quantizing-output` → `llama31_8b_base_gptq_int4` |
| INT4 transfer results | `int_4_result/` (`int4_transfer_summary.json`, `int4_transfer_bootstrap.json`) |
| Native ID results | `juice_stage2_int4_native_id/` (`int4_native_id_summary.json` also in `here_is_the/`) |
| Native ID dir_par scan | `juice_stage2_int4_native_id/directional_par_top20.json` |
| Native ID + random-K dump | `here_is_the/` (`int4_random_k_summary.json`, `jaccard_bootstrap.json`) |
| Parametric recovery dump | `here_is_the-2/` (`int4_parametric_fixes_summary.json`) |
| Kaggle notebooks | `juice_repro/kaggle_stage2_*.ipynb`, `juice_repro/kaggle_quantize_*.ipynb` |

---

## 9. One-paragraph summary (for paper / advisor)

We confirmed that GPTQ-INT4 on Llama-3.1-8B-Instruct shifts conflict arbitration (MR★ −23.8 pp on double-filtered items; corrected `follows_context` AME ~+10 pp), authorizing mechanistic follow-up. We ported JuICE's dual-run head-steering to Llama-3.1-8B and validated it at FP16 on the **base** model: context heads increase context-following by +6–10 pp and parametric heads decrease it by −6–8 pp at α=1 with full coherence (parametric clean K2 −8 pp, 95% CI [−16, −2]). Applying those **same FP16-identified heads** to a self-quantized GPTQ-INT4 base checkpoint, the same-sign transfer gate **passed** with 100% coherence. After item-level bootstrap (n=50, 1000 draws), only **context K10** is distinguishable from zero on INT4 (+12 pp, [+4, +22]). A random-K control at the same α and K is +4 pp with a CI that includes 0, so that +12 is **head-specific**, not general intervention brittleness. Native INT4 ID recovers **L20_H14, L31_H14** as directional top-2; that pair transferred at 0 pp. The raw parametric peak on both precisions is **L1_H4**; steering it at α=1 moved the **wrong way** (+6 pp, CI includes 0). Scaling the clean pair to α=3/5 produces a −12 pp drop whose CI excludes 0, but that is **`other` inflation** (25–32/50), not parametric recovery. Suppressing FP16 context K10 as H− is another −2 pp null. Parametric steering does not transfer in a usable form; context K10 does.

---

## 10. Kaggle run log — every session, final output

Rule: when a Kaggle job finishes, its summary JSON numbers go here **and** into the matching results section above. 1B mini-pilot was laptop, not Kaggle; listed first so nothing is missing.

### K0 — 1B behavioral gate (not Kaggle)

- **Where:** RTX 3050 Ti laptop
- **Output:** `mini_pilot/outputs/stats_20260824T130134Z.json`
- **Final:** MR★ Δ = **+1.7 pp**, CI includes 0. Pivot (b). Full table in `writeup/mini_pilot.md`.

### K1 — Stage 1 8B FP16 vs INT4

- **Notebook:** `mini_pilot/kaggle_llama31_8b_fp16_vs_int4.ipynb`
- **Output:** `Downloads/mini_pilot_8b_outputs/` (relabeled locally)
- **Final:** shared n=164; MR★ 46.3% → 22.6%, **Δ −23.8 pp**, CI **[−31.7, −17.1]** excludes 0; corrected AME ~+10 pp. **Go.** Details: `writeup/mini_pilot.md`.

### K2 — Self-quantize Llama-3.1-8B base → GPTQ-INT4

- **Notebook:** `juice_repro/kaggle_quantize_llama31_8b_base_gptq.ipynb`
- **Kaggle output name:** `quantizing-output` → `llama31_8b_base_gptq_int4`
- **Final:** gptqmodel 2.2, 4-bit gs128, `desc_act=True`, N_CALIB=64 smoke. This checkpoint is the INT4 model for K7–K10.

### K3 — Instruct FP16 JuICE initial reproduction

- **Notebook:** `juice_repro/kaggle_stage2_fp16_8b.ipynb`
- **Output:** user download `juice_stage2_fp16/` (not copied into this repo)
- **Final:** baseline fc **28%**; raw-ID context/parametric ~28% (weak); α=5 **collapsed** (`://` garbage). Instruct ID: enlarge Jaccard 0.54, peak ΔP ~0.008, best dir_par ~2.7×10⁻⁴.

### K4 — Instruct directional alpha sweep

- **Notebook:** `juice_repro/kaggle_stage2_cell3_alpha_sweep.ipynb`
- **Output:** `juice_stage2_alpha_sweep/alpha_sweep_summary.json`
- **Final:**
  - α+=1, α−=0: baseline 28%; context **38% (+10)**; parametric **32% (+4)**; coh 100%; P/C/O baseline 24/14/12, ctx 14/19/17, par 22/16/12
  - α+=2, α−=1: context **16% (−12)**, coh **76%**, other=38; parametric **6% (−22)**, coh **48%**, other=47

### K5 — Instruct directional Cell-3

- **Notebook:** `juice_repro/kaggle_stage2_cell3_directional.ipynb`
- **Output:** `direction-output/directional_cell3_summary.json`
- **Final (α+=1, α−=0, n=50):**

| Condition | Heads | fc | Δ | Coh | P/C/O |
|-----------|-------|----|---|-----|-------|
| baseline | — | 28% | — | 100% | 24/14/12 |
| context K10 | L22_H6, L14_H26, L11_H19, L31_H14, L11_H24, L11_H3, L11_H25, L31_H7, L2_H21, L13_H0 | 36% | **+8** | 100% | 11/18/21 |
| context K5 | first 5 of above | 34% | **+6** | 100% | 16/17/17 |
| parametric K2 | L2_H23, L8_H9 | 28% | **0** | 100% | 26/14/10 |
| parametric K3 | + L0_H2 | 26% | **−2** | 100% | 24/13/13 |

### K6 — Base model ID diagnostic

- **Notebook:** `juice_repro/kaggle_stage2_base_id_diagnostic.ipynb`
- **Output:** `juice_stage2_base_id/base_id_comparison_summary.json`
- **Final:** |D|=4, JuICE 3-type plain prompts. Raw enlarge Jaccard **0.053**; peak ΔP ctx **0.036** / par **0.058**; best dir_par **0.304**. Raw par enlarge #1 = **L1_H4**. Directional ctx top-10 starts L4_H9, L9_H0, L1_H11, L4_H5, L19_H27. Parametric directional: L20_H14 / L31_H14 / L26_H15 (clean pair L20+L26). Heads JSONs in `juice_stage2_base_id/directional_heads/`.

### K7 — Base directional Cell-3 (FP16 gate)

- **Notebook:** `juice_repro/kaggle_stage2_base_cell3_directional.ipynb`
- **Output:** `juice_stage2_base_directional_cell3/base_directional_cell3_summary.json`
- **Final (α+=1, α−=0, `juice_coherent`, n=50):**

| Condition | Heads | fc | Δ | Coh | P/C/O |
|-----------|-------|----|---|-----|-------|
| baseline | — | 16% | — | 100% | 40/8/2 |
| context K10 | L4_H9, L9_H0, L1_H11, L4_H5, L19_H27, L17_H12, L7_H26, L14_H23, L3_H22, L31_H20 | 26% | **+10** | 100% | 34/13/3 |
| context K5 | first 5 | 22% | **+6** | 100% | 36/11/3 |
| parametric K2 | L20_H14, L31_H14 | 10% | **−6** | 100% | 39/5/6 |
| parametric K3 | + L26_H15 | 10% | **−6** | 100% | 39/5/6 |
| parametric clean K2 | L20_H14, L26_H15 | 8% | **−8** | 100% | 42/4/4 |

**FP16 gate: PASSED.**

### K8 — FP16→INT4 transfer

- **Notebook:** `juice_repro/kaggle_stage2_int4_transfer.ipynb`
- **Output:** `int_4_result/int4_transfer_summary.json` + `int4_transfer_bootstrap.json`
- **Final:** same-sign gate **passed**, coherence 100% on every arm. After bootstrap, only context K10 excludes 0. Full table in §7. Baseline 14% (39/7/4). Context K10 **+12 [+4, +22]**. Clean K2 **−2 [−6, 0]** includes 0.

### K9 — Native INT4 ID + Jaccard + random-K

- **Notebook:** `juice_repro/kaggle_stage2_int4_native_id.ipynb` (ID + random-K in that session; dump `here_is_the/`)
- **Output:** `juice_stage2_int4_native_id/` and `here_is_the/int4_native_id_summary.json`, `int4_random_k_summary.json`, `jaccard_bootstrap.json`
- **Final ID:** dir par top-2 = **L20_H14, L31_H14** (J=1.0 vs FP16 noisy K2); L26_H15 rank 4; clean filter → L16_H30, L17_H27; raw par #1 = L1_H4. Ctx K10 Jaccard 0.33 (within |D|=4 noise). Full sets in §7.1.
- **Final random-K:** baseline again 14% (39/7/4), coh 100%. Random ctx K10 **+4 [−4, +12]**; random par K2 **+4 [0, +10]** (wrong way). Table in §7.1.

### K10 — INT4 parametric recovery (L1_H4 + α-grid + suppress)

- **Notebook:** `juice_repro/kaggle_stage2_int4_parametric_fixes.ipynb`
- **Output:** `here_is_the-2/int4_parametric_fixes_summary.json`
- **Final:** baseline 14% (39/7/4). **L1_H4 +6 [−2, +14]** (wrong sign; CI includes 0). Clean K2 α=0.5/1/2 = +2 / −2 / −2 (all include 0). α=3 and α=5 = **−12 [−22, −2]** excl. 0 but P/C/O **24/1/25** and **17/1/32** (`other` collapse; coherence heuristic still 100%). Suppress ctx K10 **−2 [−6, 0]**. Full table in §7.3.
