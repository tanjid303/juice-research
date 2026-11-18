# Stage 1–2 consolidated findings

**Project:** Whose Word Survives Compression?  
**This document:** locked Llama-3.1-8B through 31 Aug 2026, plus the Gemma-2-9B base Stage 2 copy through 5 Sep 2026. Skeleton for the paper’s Results + Methods.  
**Design source of truth:** `PROJECT_BRIEF_FULL.md`.

Numbers below are taken from the summary JSONs and the Stage 1 memo, not from memory. Where a figure lives only in `writeup/mini_pilot.md` (not re-derived this session), that is marked.

---

## 1. Research question

From `PROJECT_BRIEF_FULL.md`:

> If you compute JuICE’s causal intervention on the FP16 checkpoint, does it still steer the *same* model after INT4 quantization? Or did quantization change the causal object (which heads, which direction, which gain) so the FP16 recipe silently misfires — even on items where the underlying fact itself still survives quantization?

This is not “does INT4 lower QA accuracy.” The claim we are trying to earn is that quantization can transform the residual-stream geometry that source-arbitration interventions were computed in. If transfer works, the paper becomes the opposite qualification: arbitration is more robust than recall (*Compressed Lens*).

**What we have actually measured** is that question on **two families**: Llama-3.1-8B (Stage 1 Instruct behavioral + Stage 2 base identify×apply) and Gemma-2-9B **base** (Stage 2 identify×apply + GPTQ transfer only; no Stage 1 MR★). Llama numbers in §§2–4 are unchanged. Gemma is §7.

---



## 2. Stage 1 — Behavioral findings (Llama-3.1-8B-Instruct)

**Setup (one paragraph).** 250 ConflictQA popular items (`data/conflictqa_pilot250_popular.jsonl`). FP16 `meta-llama/Llama-3.1-8B-Instruct` vs third-party GPTQ-INT4 `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` (4-bit, gs128). Greedy decode, T=0. Double-filter: keep items closed-book correct at **both** precisions → **n = 164**. Hardware: Kaggle 2× T4; INT4 via gptqmodel `BACKEND.TORCH`. Full memo: `writeup/mini_pilot.md`.

### 2.1 MR★ (primary behavioral estimand)


| Precision | Closed-book     | MR (unfiltered) | MR★ (shared 164)   |
| --------- | --------------- | --------------- | ------------------ |
| FP16      | 182/250 (72.8%) | 92/250 (36.8%)  | **76/164 (46.3%)** |
| INT4      | 175/250 (70.0%) | 48/250 (19.2%)  | **37/164 (22.6%)** |


**Δ MR★ (INT4 − FP16) = −39/164 = −23.8 pp.**  
Bootstrap 95% CI (seed 42, 1000 resamples of the 164-item set): **[−31.7, −17.1] pp. Excludes 0.**

Closed-book is nearly stable (−7 items). Unfiltered MR also drops (−17.6 pp), so this is not “pure forgetting that vanishes after the double filter.”

### 2.2 `follows_context` AME (secondary)

`follows_context := conflict_label == "context"` (excludes `rejects_context`). Same 164 items. **Preferred published number: cluster-robust logit AME.**

*Source:* `writeup/mini_pilot.md` *(relabeled stats; mixed-model objects are not in this repo). Not re-fit this session.*


| Estimand                 | Point        | 95% CI            | excl. 0? |
| ------------------------ | ------------ | ----------------- | -------- |
| Cluster-robust logit AME | **+10.3 pp** | **[+3.1, +17.5]** | yes      |
| Raw Δ P(context)         | +10.4 pp     | (52→69)/164       | —        |
| Item-averaged AME        | +9.8 pp      | —                 | —        |
| Mixed VB AME             | +16.1 pp     | [+5.9, +26.2]     | yes      |


Publish **+10.3 pp [+3.1, +17.5]** (cluster-robust). Mixed VB is directionally consistent but larger; do not lead with it.

### 2.3 Label shift and transitions (shared 164, corrected scorer)


| Label           | FP16 | INT4 | Δ       |
| --------------- | ---- | ---- | ------- |
| parametric      | 76   | 37   | **−39** |
| context         | 52   | 69   | **+17** |
| rejects_context | 13   | 22   | +9      |
| other           | 23   | 36   | +13     |


**FP16 → INT4 transitions:**


|                       | → parametric | → context | → rejects_context | → other |
| --------------------- | ------------ | --------- | ----------------- | ------- |
| **parametric →**      | 33           | **20**    | 10                | 13      |
| **context →**         | 3            | 41        | 5                 | 3       |
| **rejects_context →** | 0            | 3         | 5                 | 5       |
| **other →**           | 1            | 5         | 2                 | 15      |


Dominant mass off parametric: 20 → context, 10 → rejects_context, 13 → other. Unlike the 1B screen (parametric flat, context → `other` via brevity), 8B is a real arbitration shift: parametric collapses, context rises.

### 2.4 Scorer-bug history (short)

The first AME pass treated negation/denial that *named* the context entity as `context` (substring match). About 10/26 of the pre-fix `parametric→context` flips were rejections (e.g. “Sitten is not the capital of Lebanon”). Pre-fix mixed AME was **+20.6 pp** (inflated). Fix: denial frames → `rejects_context`; bare entity names matching context gold → `context`. Relabel only, no re-inference. All later numbers in this section use the corrected scorer. Keep `rejects_context` in breakdowns; AME still uses `== "context"` only.

### 2.5 What Stage 1 does *not* claim

- 1B (Llama-3.2-1B-Instruct, n★=115) was a **null / confound screen** (MR★ +1.7 pp, CI includes 0). Do not mix it into the 8B claim.
- Stage 1 INT4 is a **third-party Instruct GPTQ**. Stage 2 mechanistic cells use a **self-quantized base** checkpoint. Do not treat them as the same weight file.

**Stage 1 decision:** §5.1 gate passed. Proceed to Stage 2 JuICE on 8B.

---



## 3. Stage 2 — Mechanistic findings (Llama-3.1-8B **base**)

Intervention: JuICE dual-run (record per-head post-`o_proj` contributions, inject `α × saved` on a second forward). Port in `common/juice_attn.py` / `common/juice.py`. Eval: 50 held-out ConflictQA items (`split.json`, seed 42), `prompt_format="juice_coherent"`, greedy, `max_new_tokens=16`. Metric: `follows_context`. Paired bootstrap: resample the 50 items with replacement, 1000 draws, seed 42, percentile 95% CI on Δ = steered − baseline (paired on item id). Grid is 2 pp (1/50).

### 3.1 Why Instruct → base

JuICE’s original models are **base**, not Instruct (Li et al., ICML 2025). On Llama-3.1-8B-Instruct + chat template, parametric identification and steering failed:


|                                         | Instruct (chat)       | Base (plain JuICE prompts) |
| --------------------------------------- | --------------------- | -------------------------- |
| Raw enlarge Jaccard (ctx vs par H+)     | **0.54**              | **0.053**                  |
| Peak raw `scores_pos` (ctx / par)       | 0.0080 / 0.0077       | **0.036 / 0.058**          |
| Best `directional_par`                  | **2.68×10⁻⁴**         | **0.304** (~1130×)         |
| Directional top-10 Jaccard (ctx vs par) | 0                     | 0                          |
| Parametric Cell-3 at α+=1               | 0 / −2 pp (K=2 / K=3) | **−6 to −8 pp**            |
| Context Cell-3 at α+=1                  | +6 to +8 pp           | +6 to +10 pp               |


Sources: `juice_stage2_base_id/base_id_comparison_summary.json` (`instruct_reference` vs base); Instruct Cell-3 `direction-output/directional_cell3_summary.json`; base Cell-3 `juice_stage2_base_directional_cell3/base_directional_cell3_summary.json`.

Chat-tuning **suppresses the parametric-directional ID signal** (~1100×) and leaves parametric steering near zero, while context steering still works. That is a model+prompt confound, not evidence that the dual-run port is broken. **All INT4 transfer cells use base + base-identified heads.**

**FP16 base gate (must pass before transfer):**


| Condition           | Heads                  | fc  | Δ       | Coh  | P/C/O   |
| ------------------- | ---------------------- | --- | ------- | ---- | ------- |
| baseline            | —                      | 16% | —       | 100% | 40/8/2  |
| context K10         | directional ctx top-10 | 26% | **+10** | 100% | 34/13/3 |
| context K5          | top-5 of those         | 22% | +6      | 100% | 36/11/3 |
| parametric K2       | L20_H14, L31_H14       | 10% | −6      | 100% | 39/5/6  |
| parametric K3       | + L26_H15              | 10% | −6      | 100% | 39/5/6  |
| parametric clean K2 | **L20_H14, L26_H15**   | 8%  | **−8**  | 100% | 42/4/4  |


FP16 bootstrap (same 50 items, 1000 draws): context K10 **+10 [+2, +20]** excludes 0; clean K2 **−8 [−16, −2]** excludes 0. Source: `int_4_result/int4_transfer_bootstrap.json` → `fp16`.

Canonical heads locked for transfer: context K10/K5 from directional rerank; parametric **clean K=2 = L20_H14, L26_H15** (L31_H14 is context-suppression-heavy: FP16 `par_pos` = −0.112).

### 3.2 Context K10 transfers, and is head-specific

INT4 checkpoint (**canonical**): self-quantized `llama31_8b_base_gptq_int4_ncalib128` (gptqmodel 2.2, 4-bit, gs128, `sym=True`, `desc_act=True`, **N_CALIB=128**). Same 50 eval items, α+=1, α−=0, FP16 heads, no re-ID. Source: `this_is_the_128/int4_transfer_summary.json` + `this_is_the_128/int4_ncalib128_transfer_bootstrap.json`.


| Condition       | INT4 fc | INT4 Δ [95% CI]   | excl. 0?              | FP16 Δ [CI]   | Sign vs FP16 | Coh  | P/C/O   |
| --------------- | ------- | ----------------- | --------------------- | ------------- | ------------ | ---- | ------- |
| baseline        | 20%     | —                 | —                     | 16%           | —            | 100% | 39/10/1 |
| **context K10** | 38%     | **+18 [+8, +30]** | **yes** (1000/1000 +) | +10 [+2, +20] | same         | 100% | 28/19/3 |


**Random-K control** (same α, same K, heads drawn excluding the FP16 identified set; seed 42 for K10). Source: `this_is_the_128/int4_random_k_summary.json`. Baseline in that session 20% (39/10/1), coh 100%. Random heads are the same draw as the N=64 run.


|               | Identified                | Random-K                   |
| ------------- | ------------------------- | -------------------------- |
| context K10 Δ | **+18 [+8, +30]** excl. 0 | **−6 [−14, 0]** includes 0 |


The +18 is **head-specific** by the same detection rule as before: identified CI excludes 0, random-K CI includes 0. CIs still overlap at n=50 (control-fails / treatment-detects, not a proven difference-of-deltas). Random-K coherence 100% — not a collapse control.

Context K5 was not re-run at N_CALIB=128. On the N=64 checkpoint it matched FP16 in point estimate (+6) but neither CI excluded 0 (`int_4_result/int4_transfer_bootstrap.json`: +6 [−2, +16]; P/C/O 37/10/3, coh 100%).

**Calibration robustness (N_CALIB=64 vs 128).** The earlier self-quant `llama31_8b_base_gptq_int4` (**N_CALIB=64**) is a robustness check, not the headline. Same 50 items, same locked heads, same seeds. Sources: `int_4_result/int4_transfer_bootstrap.json`, `here_is_the/int4_random_k_summary.json`, `this_is_the_128/n64_vs_n128_comparison.json`.


| Condition           | N64 Δ [CI]                | N128 Δ [CI]               | same sign? | excludes-0 flip? |
| ------------------- | ------------------------- | ------------------------- | ---------- | ---------------- |
| context K10         | **+12 [+4, +22]** excl. 0 | **+18 [+8, +30]** excl. 0 | yes        | no               |
| random-K10          | +4 [−4, +12] incl. 0      | −6 [−14, 0] incl. 0       | no         | no               |
| parametric clean K2 | −2 [−6, 0] incl. 0        | −4 [−10, 0] incl. 0       | yes        | no               |


No detection verdict flipped. Context K10 remains the only detected INT4 effect at both calib sizes. Random-K10’s **point estimate** changes sign (+4 → −6); both CIs include 0 in both runs. The control is noisy at n=50, and detection is the CI-inclusion call, not the point estimate. Do not read the sign flip as a stronger contrast.

N=64 random-K session: baseline 14% (39/7/4), identified +12 [+4, +22], random +4 [−4, +12], coh 100%.

**Pre-registered same-sign gate:** passed on N=64 (context K10/K5 and parametric clean K2 keep sign; coherence ≥ 0.90). N=128 re-ran context K10 and clean K2 only; both keep sign vs FP16, coherence 100%. After CIs, the **only detected INT4 effect is context K10**.

**Bidirectionality check: reverse transfer and native in-precision identification.** Same 50 items, α+=1, α−=0 unless noted, bootstrap 1000 / seed 42. Identify on the native INT4 directional context K10 (not the FP16 set): `L7_H26, L14_H10, L7_H21, L4_H5, L3_H3, L31_H20, L4_H11, L14_H23, L8_H29, L9_H0`. Jaccard vs FP16 K10 = **0.33** (overlap 5/15: `L7_H26, L4_H5, L31_H20, L14_H23, L9_H0`). Baselines reused from prior sessions (FP16 fc 16% P/C/O 40/8/2; INT4 N128 fc 20% P/C/O 39/10/1). Source: `unga-bunga-23/juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_summary.json` + `int4_reverse_and_native_k10_bootstrap.json`.


| Cell | Identify → apply | fc | Δ [95% CI] | excl. 0? | Coh | P/C/O | Comparator |
| ---- | ---------------- | -- | ---------- | -------- | --- | ----- | ---------- |
| **A reverse** | INT4 K10 → FP16 | 64% | **+48 [+34, +62]** | **yes** (1000/1000 +) | 100% | 12/32/6 | FP16-ID K10 on FP16: **+10 [+2, +20]** |
| **B native** | INT4 K10 → INT4 N128 | 38% | **+18 [+8, +30]** | **yes** (1000/1000 +) | 100% | 24/19/7 | FP16-ID K10 on INT4 N128: **+18 [+8, +30]** |


**Cell B is confirmed.** Native INT4 identification + native INT4 application reproduces the forward-transfer magnitude (**+18 [+8, +30]** on both). Manual inspection rules out a Cell A–style scorer trap: exam-meta **0/19** among context labels (0/50 any label); 16/19 same context ids as forward transfer; 11/19 identical text; 40/50 same label. The 7 `other` labels are alias/brevity misses (`Lahore` vs gold “The Indomitable City”; bare `Kangra` / `Osona` / `Yungay Province`), not format collapse. Completions are on-topic context-entity names in the same style as the trusted FP16→INT4 run. Source: `unga-bunga-23/cell_b_diagnostics.json`, `unga-bunga-23/cell_b_inspection.md`.

That is an independent triangulation of the context K10 effect: FP16’s own K10, FP16 heads transferred onto INT4, and INT4’s own K10 applied in-precision all move `follows_context` the same way, and two of the three have been read as text.

**Cell A is a rejected same-circuit hypothesis.** The scorer records a real, α-driven positive (**+48 [+34, +62]** at α=1). Manual inspection shows a different behavioral object than FP16’s own K10 or the forward-transfer result: 13/32 context labels at α=1 are bare context-entity names followed by exam-meta (“Edina / Rumbek / … Note: The question is a multiple-choice question…”). Letter-heuristic coherence stays 100%. Source: `unga-bunga-23/cell_a_diagnostics.json`, `unga-bunga-23/cell_a_inspection.md`.

The failure is in **cross-precision application**, not in INT4 identification quality. On INT4’s own ID, 9/10 of the native K10 heads are context-specific (`ctx_pos>0` and `par_pos≤0`); `L8_H29` is the exception (`par_pos` +0.0009). That is not a universal-booster ID signature (FP16’s own K10 is 0/10 both-positive on FP16 ID). Two INT4-only heads are context-**negative** when scored on FP16 ID: `L7_H21` `ctx_pos` **−0.0058** (INT4 ID **+0.0120**), `L3_H3` `ctx_pos` **−0.0185** (INT4 ID **+0.0217**). Steering that set on FP16 is not “the same K10.”

α-sweep on FP16, same heads, reused baseline (fc 16%, P/C/O 40/8/2). Source: `resultofthe/juice_stage2_int4_ncalib128/cell_a_alpha_sweep_summary.json`.


| α+ | fc | Δ [95% CI] | excl. 0? | Coh | exam-meta | P/C/O |
| -- | -- | ---------- | -------- | --- | --------- | ----- |
| 0.25 | 50% | **+34 [+20, +50]** | **yes** (1000/1000 +) | 100% | **58%** | 21/25/4 |
| 0.5 | 52% | **+36 [+22, +50]** | **yes** (1000/1000 +) | 100% | **52%** | 18/26/6 |
| 1.0 | 64% | **+48 [+34, +62]** | **yes** (1000/1000 +) | 100% | 36% | 12/32/6 |


A scaled copy of FP16’s own K10 (+10 [+2, +20]) should have dropped toward ~+10 at α=0.25/0.5 and lost the exam-meta dump. It did not. The floor is **+34 pp** with *more* exam-meta at low α, not less. `other` stays 4–6 (not the α=3/5 parametric collapse). Δ still rises with α, so something is being driven — format/scorer collapse, not genuine context-arbitration. We are **not** claiming symmetric bidirectional transfer. Context-following steering is robust *within* each precision (FP16 own K10; INT4 native K10; FP16→INT4 forward transfer) and does **not** reverse-transfer cleanly (INT4→FP16). Same discipline as the 1B brevity confound and the α=3/5 trap: scorer-level detection is necessary, not sufficient.

**Methods footnote (brevity / parametric-first scorer).** On both Cell B and the trusted forward-transfer cell, parametric→context flips include cases where the unsteered INT4 baseline already named the context entity in a full sentence (`Rumbek is the capital of Wisconsin.`), labeled `parametric` because `answers_match` checks parametric first, and steering shortens it to a bare name (`Rumbek`) that then labels `context`. All 9 Cell B flips are of this kind. This is a pipeline property of `common/scoring.py` on `juice_coherent` completions, not a Cell-B-specific artifact. It applies wherever this scorer meets context-K10 shortening. Do not re-litigate it as a second-family blocker; do report label counts and read a sample of texts.

### 3.3 Parametric direction: complete negative

FP16 clean K2 is a real effect (−8 pp, CI excludes 0). On INT4, every path we tested fails to recover a usable parametric steer. Sources: `this_is_the_128/int4_ncalib128_transfer_bootstrap.json` (canonical N_CALIB=128 transfer), `int_4_result/int4_transfer_bootstrap.json` (N=64 check), `here_is_the-2/int4_parametric_fixes_summary.json`, `here_is_the/int4_native_id_summary.json`, `ungabunga_result/juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json` (native clean pair, applied on the N_CALIB=128 checkpoint). Recovery arms below (α-grid, L1_H4, native directional K2/K3, H−) were run on the N=64 checkpoint and were not repeated at N=128. The native *clean* pair is the exception: identified on N=64, applied on N=128.

**0. Transfer of the FP16 clean pair (the starting null).**  
L20_H14 + L26_H15, α+=1, α−=0.

Canonical (N_CALIB=128): **−4 pp [−10, 0]**, includes 0. 858/1000 draws negative, **142 zero, 0 positive**. Coherence 100%, P/C/O 37/8/5. Source: `this_is_the_128/int4_ncalib128_transfer_bootstrap.json` + `this_is_the_128/int4_transfer_summary.json`. Compatible with a true delta of 0. Not a sign flip, not a collapse, **not a detected effect**.

Calibration check (N_CALIB=64): **−2 pp [−6, 0]**, includes 0. 622/1000 negative, **378 zero, 0 positive**. Coh 100%, P/C/O 41/6/3. Same locked pair, same 50 items. The parametric null is **not an artifact of N_CALIB=64**.

Related transfer arms (N=64 cell; not re-run at N=128): noisy K2 L20_H14+L31_H14 = **0 [−10, +10]** (sign *flip* vs FP16 −6); K3 = −4 [−12, +6] includes 0.

**1. Alpha grid on the same clean pair (rescale repair).**  
α+ ∈ {0.5, 1, 2, 3, 5}, α−=0, n=50.


| α+  | Δ [95% CI]        | excl. 0? | P/C/O       | Coh heuristic |
| --- | ----------------- | -------- | ----------- | ------------- |
| 0.5 | +2 [0, +8]        | no       | 39/8/3      | 100%          |
| 1   | −2 [−6, 0]        | no       | 41/6/3      | 100%          |
| 2   | −2 [−12, +8]      | no       | 35/6/9      | 100%          |
| 3   | **−12 [−22, −2]** | **yes**  | **24/1/25** | 100%*         |
| 5   | **−12 [−22, −2]** | **yes**  | **17/1/32** | 100%*         |


Letter-based `looks_coherent` (no `://` spam). At α=3/5, `other` **is 50–64%**: fluent-looking rambling / blank-slot / off-entity completions, not parametric answers. Parametric labels fall 39→24→17; context does not convert to parametric. **Do not read −12 as recovered parametric steering.** Rescale does not restore the FP16 −8 pp object.

**2. L1_H4, the raw parametric peak (different-head repair).**  
Only INT4 (and FP16) head with `par_pos` above 5× the ID noise floor. Steer K=1, α+=1, α−=0: **+6 [−2, +14]**, includes 0. **Wrong sign** for a parametric enlarge (more context-following). 882/1000 draws positive. Coh 100%, P/C/O 36/10/4. The “same raw peak, just steer that head” story is rejected.

**3. Native INT4 re-identification.**  
Same protocol as FP16 base: |D|=4 (same four calib IDs), 3-type JuICE prompts, FAST alphas.


| Spec                                   | INT4 set                | vs FP16                     | Apply result                                             |
| -------------------------------------- | ----------------------- | --------------------------- | -------------------------------------------------------- |
| directional par K2                     | **L20_H14, L31_H14**    | Jaccard **1.0** (same pair) | already applied in transfer: **0 [−10, +10]**            |
| directional par K3                     | L20_H14, L31_H14, L8_H9 | J=0.50                      | K3 transfer −4 [−12, +6]                                 |
| “clean” K2 (`par_pos>0` ∧ `dir_par>0`) | **L16_H30, L17_H27**    | Jaccard **0.00**            | **−2 [−8, +4]** includes 0; coh 100%; P/C/O **39/9/2** |


Native directional top-2 is **not a different pair** — it is FP16’s noisy K2, which does not steer on INT4. Native clean filter *does* return a different pair. That pair was applied on the N_CALIB=128 checkpoint (K=2, α+=1, α−=0, same 50 items, seed 42). Source: `ungabunga_result/juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json` + `int4_native_clean_k2_bootstrap.json`.

Canonical native-clean apply: **−2 pp [−8, +4]**, includes 0. 622/1000 draws negative, 214 zero, **164 positive**. Coherence 100%. Parametric count is **unchanged at 39**; context 10→9, other 1→2. In-session baseline matched the N=128 transfer session item-for-item (`identical: true`, fc 20%). No FP16 same-sign gate applies (Jaccard 0.00 vs L20_H14/L26_H15). This is a null, not a repair: near-noise `par_pos` (+0.020 / +0.015) did not become a usable parametric steer. Random parametric K2 moves the **wrong way** (+4 [0, +10]).

**4. Context-suppression as a parametric proxy.**  
H− = FP16 directional context K10, enlarge empty, α+=0, α−=1: **−2 [−6, 0]**, includes 0. P/C/O 44/6/0. Same one-item nudge as clean K2 at α=1.

**Jaccard vs |D|=4 noise** (25 calib-item resamples; `here_is_the/jaccard_bootstrap.json`). Observed FP16-vs-INT4 overlap is **within** the INT4-vs-INT4 noise floor for every spec we computed (ctx K10 J=0.33 vs noise mean 0.27 [0.05, 0.82]; par K2 J=1.0 vs noise already ~0.95). Head-set drift is not identifiable at this calibration size except as “top-2 parametric is stable.”

### 3.4 Identification methodology: directional re-ranking has a blind spot

`directional_par := par_pos − ctx_pos`. Heads that **suppress context tokens a lot** (large negative `ctx_pos`) can outrank heads that actually **boost parametric tokens**.


| Head      | Role                                             | FP16 `par_pos` / `ctx_pos` / dir_par (rank) | INT4 `par_pos` / `ctx_pos` / dir_par (rank) |
| --------- | ------------------------------------------------ | ------------------------------------------- | ------------------------------------------- |
| **L1_H4** | raw parametric enlarge **#1** on both precisions | **+0.0577** / −0.0392 / +0.0969 (**#29**)   | **+0.0578** / −0.0581 / +0.1159 (**#14**)   |
| L20_H14   | directional #1; in clean + noisy K2              | +0.0072 / −0.297 / **+0.304 (#1)**          | **−0.0436** / −0.301 / +0.257 (**#1**)      |
| L31_H14   | noisy K2 (ctx-suppression)                       | −0.112 / −0.378 / +0.267 (#2)               | −0.104 / −0.360 / +0.256 (#2)               |
| L26_H15   | clean K2                                         | +0.0184 / −0.205 / +0.223 (#3)              | −0.0019 / −0.185 / +0.183 (#4)              |


Sources: `juice_stage2_base_id/id_diagnostics_raw.json` + `directional_rerank_base.json`; `juice_stage2_int4_native_id/id_diagnostics_raw.json` + `directional_par_top20.json`. INT4 noise proxy (median max(|ctx|,|par|)) = 9.2×10⁻³; 5× threshold = 0.046. Only L1_H4 in/near top-20 clears that threshold with **positive** `par_pos`.

L1_H4 is **not** an INT4-only discovery: `par_pos` matches FP16 to three digits. Cell-3/transfer never used it because directional ranking preferred universal context-suppressors. When we did steer it, it did not produce a parametric effect (wrong sign, §3.3). The blind spot is still a methods finding: **do not treat directional top-K as “the parametric heads” without checking the sign of** `par_pos`**.**

---



## 4. What this means so far

Quantization does not uniformly break this causal mechanism — it is direction-selective. Context-following circuitry survives GPTQ INT4 quantization intact and head-specifically. The context K10 effect is now triangulated three ways: FP16’s own K10 (**+10 [+2, +20]**), FP16 heads transferred onto INT4 (**+18 [+8, +30]**), and native INT4 identification + application (**+18 [+8, +30]**). Manual inspection on the two INT4 cells that share that +18 (forward transfer and Cell B) shows the same fluent / short-name style and **0–5%** exam-meta among context labels. Reverse transfer (INT4 K10 → FP16) is a large scorer-level positive that is **not** that object: exam-meta 36–58%, α-sweep floor +34 pp, two INT4-only heads context-negative on FP16 ID. Transfer here is directional, not symmetric — INT4-identified heads carry precision-specific geometry that does not map back cleanly. Parametric-memory steering does not survive in any recoverable form we tested.

That is the 8B-base result, not a claim about all LLMs. The same-sign transfer gate still *passed* (we did not observe a context sign-flip or a coherence collapse at α=1). After CIs, recovery attempts, and the reverse-transfer diagnostics, the mechanistic sentence that survives is narrower than the pre-registered gate: **context K10 is a real, head-specific, precision-transferable INT4 effect in the forward direction and in-precision; parametric JuICE is not a detected INT4 effect at this n, and rescale / raw-peak / native re-ID (including the native clean pair L16_H30/L17_H27) / context-H− do not repair it.** We do not claim bidirectional transfer of the same circuit.

Those two core claims are robust to GPTQ calibration-set size. Re-quantizing the same base model at **N_CALIB=128** (vs N_CALIB=64) and re-running the three locked cells on the same 50 items / same heads / same seeds did not flip any “excludes 0” call: context K10 stays detected, random-K10 and parametric clean K2 stay nulls. That is evidence against “the smaller calib set produced a spurious transfer result,” not a new mechanistic finding.

Relative to the brief’s A/B/C repair hypotheses: this is **not** a clean Hypothesis B (rescale failed without producing parametric answers). It is **not** a clean Hypothesis A for parametric heads: native directional top-2 is the *same* pair and still does not steer, and the native clean pair (a *different* pair) is also a null. Context looks closer to “same object, still works” **in-precision and FP16→INT4**, independently recovered by native INT4 re-ID. Reverse transfer is the same *sign* on the scorer and a different object in the text. Parametric looks like the causal object we can actually *use* did not survive in usable form — including the raw-ID peak, which ID finds on both precisions but which does not steer parametric on INT4.

---



## 5. What the second family had to establish (and what Gemma did)

Llama left five checks for a second family. Gemma-2-9B **base** ran Stage 2 identify×apply + GPTQ transfer on the same 50 eval IDs, same JuICE port, same GPTQ recipe (N_CALIB=128). Full tables: §7. Llama numbers in §§2–4 are unchanged.

1. **Asymmetry does not replicate as Llama’s pattern.** Llama: context K10 enlarge transfers head-specifically; parametric clean does not. Gemma: directional context K10 **enlarge** is a FP16 null (48% baseline is already a ceiling). The object that transfers head-specifically is the **raw parametric peak L27_H6**, not context enlarge.
2. **Parametric transfer is not “GPTQ always kills memory steering.”** L27_H6, K=1, α+=1: FP16 **−12 [−22, −4]** excl. 0; INT4 **−8 [−16, −2]** excl. 0, same sign; random-K1 **0 [0, 0]**. Directional/clean K2 (L31_H7, L39_H6) is a FP16 null — those heads are ID suppressors, not a Llama-style clean pair.
3. **Directional blind spot is real on Gemma, and this time the raw peak works.** Raw `par_pos` #1 = L27_H6; directional par top-2 = L31_H7 / L39_H6 (`par_pos` at/below noise). Steering the raw peak is the detected parametric object. Llama’s L1_H4 analog was the opposite behavioral outcome (wrong-sign / null on INT4).
4. **Checkpoint discipline.** Gemma Stage 2 INT4 is self-quant `gemma2_9b_base_gptq_int4_ncalib128` (same recipe as Llama N128). There is **no** Gemma Stage 1 Instruct GPTQ in this document.
5. **Forward-only / reverse transfer** was not run on Gemma. Native INT4 re-ID was not run. Do not fill those cells from Llama.

**Limitations (Llama, unchanged):** unsteered INT4 `follows_context` shifted **14% → 20%** between N=64 and N=128 on Llama (same 50 ids). Gemma INT4 baseline was **46%** vs FP16 **48%** on the same 50 items (L27_H6 session).

---



## 6. Methods summary (copy-adapt for the paper)



### 6.1 Models and quantization


| Role                                               | Model                     | Precision | Checkpoint                                                                       |
| -------------------------------------------------- | ------------------------- | --------- | -------------------------------------------------------------------------------- |
| Stage 1 behavioral                                 | Llama-3.1-8B-**Instruct** | FP16      | `meta-llama/Llama-3.1-8B-Instruct`                                               |
| Stage 1 behavioral                                 | same                      | GPTQ-INT4 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` (third-party)              |
| Stage 2 ID + steer                                 | Llama-3.1-8B **base**     | FP16      | `meta-llama/Llama-3.1-8B`                                                        |
| Stage 2 transfer (headline)                        | same                      | GPTQ-INT4 | self-quantized `llama31_8b_base_gptq_int4_ncalib128` (**N_CALIB=128**)           |
| Stage 2 transfer robustness / native ID / recovery | same                      | GPTQ-INT4 | self-quantized `llama31_8b_base_gptq_int4` (N_CALIB=64; retained for comparison) |
| Stage 2 ID + steer                                 | Gemma-2-9B **base**       | FP16      | `google/gemma-2-9b`                                                              |
| Stage 2 transfer                                   | same                      | GPTQ-INT4 | self-quantized `gemma2_9b_base_gptq_int4_ncalib128` (**N_CALIB=128**)             |


**Self-quant recipe (Stage 2 INT4):** gptqmodel 2.2, transformers 4.51.3, 4-bit, `group_size=128`, `sym=True`, `desc_act=True`, C4 calib truncated to 512 tokens, **N_CALIB = 128 (canonical)**. Identical recipe at N_CALIB=64 kept as a robustness check on Llama only (`llama31_8b_base_gptq_int4`). Load: gptqmodel `BACKEND.TORCH` on T4 (Marlin JIT failed on sm_75). Decode: greedy, T=0, batch 1. Gemma uses the same recipe; checkpoint `gemma2_9b_base_gptq_int4_ncalib128`. **Do not** force `torch_dtype=float16` when loading the Gemma GPTQ dump (weights are bf16; a float16 override doubles RAM and SIGKILLs). Gemma FP16 load: `attn_implementation="eager"`.

**Flag:** Stage 1 and Stage 2 INT4 weights are **different files** (Instruct community GPTQ vs self-quantized base). Paper text must not imply a single INT4 artifact. Gemma has **no** Stage 1 Instruct GPTQ in this document.

### 6.2 Data and splits

- Data: `data/conflictqa_pilot250_popular.jsonl` (ConflictQA popular, n=250).
- Stage 1 analysis set: double-filter closed-book correct at FP16 **and** INT4 → **n = 164**.
- Stage 2 eval: **n = 50** held-out IDs in `juice_stage2_base_id/directional_heads/split.json` (seed 42). The same 50 items are used for Llama FP16 Cell-3, INT4 transfer (N=64 and N=128), random-K, and parametric recovery, **and** every Gemma Stage 2 cell (`juice_stage2_gemma2_base_id/directional_heads/split.json` is the same seed-42 split).
- Head ID calibration: **|D| = 4** items (first four `calib_ids` in `split.json`; the JSON also lists 25 calib IDs, but ID runs used 4). Same four IDs on FP16 and INT4. Prompt family: JuICE 3-type plain completions — `clean`, `substitution_conflict`, `coherent_conflict` — no chat template on base. ID scan alphas: `FAST_POS_ALPHAS = (1, 3, 5)`, `FAST_NEG_ALPHAS = (−1, −2)`.



### 6.3 Intervention and scoring

- Dual-run JuICE: pass 1 records per-head outputs after `o_proj`; pass 2 adds `α × saved` to `o_proj` output. Two full forwards per decode token. Llama wrap: `GetHeadsLlamaAttention`. Gemma-2 wrap: `GetHeadsGemma2Attention` (GQA 16/8, `head_dim=256`, attn logit softcap 50).
- K: context K=10 (JuICE 7–8B convention) and K=5; parametric K=2 (clean) / K=2 noisy / K=3. Enlarge-only unless noted (α−=0). Gemma also ran K=1 on the raw par peak (L27_H6) and H− suppress (α+=0, α−=1) on ctx K10 and the locked random-K10.
- Default steering gain for transfer: **α+ = 1, α− = 0**. Instruct sweep showed α+=2 / α−=1 collapses coherence (ctx coh 76%, par coh 48%). Recovery grid additionally used α+ ∈ {0.5, 2, 3, 5} and one H− run (α+=0, α−=1 on FP16 context K10). Gemma suppress cells use that same H− recipe.
- Eval prompt (base Cell-3 / INT4): `juice_coherent` (coherent-conflict completion).
- Labels: `{parametric, context, rejects_context, other}` via `common/scoring.py`. `follows_context` = `label == "context"`.
- Coherence heuristic: reject `://` spam, all-punct, repeated-token garbage; require alphabetic content. **This heuristic stayed at 100% while α=3/5** `other` **hit 50–64%.** Report label counts, not only the heuristic. Gemma Stage 2 used the letter-based `looks_coherent` rule; every reported Gemma cell is 100%.



### 6.4 Statistics

- **Stage 1 MR★:** paired bootstrap on the 164 double-filtered items, 1000 draws, seed 42. Report pp difference and percentile 95% CI.
- **Stage 1 AME:** logistic mixed-effects `follows_context ~ precision + (1|item)` and cluster-robust logit AME on the same 164 items. Headline: cluster-robust **+10.3 pp [+3.1, +17.5]**.
- **Stage 2 deltas:** paired bootstrap on the **50 eval items**, 1000 draws, seed 42. Δ = steered `follows_context` − baseline, paired on item id. Percentile 95% CI. “Excludes 0” is the detection rule used in the text.
- **Random-K:** same n, α, K; heads sampled excluding the FP16 identified set (seeds 42–45 for ctx K10 / K5 / par K2 / K3). Gemma locked one seed-42 random-K10 (excluding ctx K10 ∪ par K2 ∪ L27_H6) and reused it as enlarge control and as H− control; Gemma random-K1 for L27_H6 is L40_H15.
- **Jaccard noise floor:** resample |D|=4 calib items with replacement, 25 draws, re-rank from saved per-item score maps; compare observed FP16-vs-INT4 Jaccard to INT4-vs-INT4.
- **Gemma suppress specificity / DiD:** paired item-level bootstrap, 1000 draws, seed 42. Specificity = Δ_identified − Δ_random; DiD = spec_INT4 − spec_FP16. Do not infer identified-vs-random from overlapping cell CIs.



### 6.5 Pre-registered gate vs what we report

Gate (`PROJECT_BRIEF_FULL.md` §5.5 / transfer notebook): primary conditions (context K10/K5, parametric clean K2) keep **the same sign** as FP16 and coherence ≥ 0.90. Magnitude match was not required. **Gate passed** on Llama.  
After CIs + recovery + reverse-transfer diagnostics: report **context K10 as the detected INT4 effect** (forward transfer and native apply; not INT4→FP16 reverse transfer); report parametric as a **null at α=1** plus failed repairs, not as “weaker but real attenuation.”

On Gemma the Llama gate objects do not map 1:1. Report **L27_H6 K=1 as the detected transferable object** (FP16 and INT4; random-K1 is a hard zero). Report directional ctx K10 enlarge and directional par K2 as FP16 nulls. Report ctx K10 suppress as a detected FP16 effect whose INT4 identified-vs-random contrast is **not** detected (DiD includes 0). Native INT4 re-ID and reverse transfer were not run.

---


## 7. Second family — Gemma-2-9B base (Stage 2)

**Setup.** `google/gemma-2-9b` **base** (not Instruct). Same 50 eval IDs as Llama (`split.json` seed 42), same four calib IDs for ID, `juice_coherent`, greedy, dual-run JuICE, paired bootstrap 1000 / seed 42, percentile 95% CI, detection = CI excludes 0. Coherence = letter-based `looks_coherent` (100% on every Gemma cell below). Wrap: `GetHeadsGemma2Attention` (GQA 16/8, `head_dim=256`, attn logit softcap 50). FP16 load uses `attn_implementation="eager"`. INT4: self-quant `gemma2_9b_base_gptq_int4_ncalib128` (gptqmodel 2.2, transformers 4.51.3, 4-bit, gs128, `sym=True`, `desc_act=True`, C4 512 tokens, **N_CALIB=128**, load `BACKEND.TORCH`). Do not requantize. Do not force `torch_dtype=float16` on the GPTQ load (bf16 dump; 2× RAM SIGKILL).

**Not run on Gemma:** Stage 1 MR★ / Instruct behavioral, native INT4 re-ID, reverse transfer (INT4 heads → FP16). Those cells stay empty.

### 7.1 Identification (FP16)

Source: `juice_stage2_gemma2_base_id/base_id_comparison_summary.json`. |D|=4 (same calib IDs as Llama). Noise-proxy median **0.0063**. Directional ctx vs par top-10 Jaccard **0.0**.

**Directional context K10:** L40_H14, L17_H7, L24_H15, L22_H0, L32_H11, L25_H0, L3_H9, L8_H9, L7_H6, L29_H5.

**Directional / clean parametric K2:** **L31_H7, L39_H6** (identical). These are context **suppressors**: `ctx_pos` −0.115 / −0.094; `par_pos` 0.002 / 0.010 (at/near the noise floor). They rank as “parametric” only because directional_par = par_pos − ctx_pos is large when ctx_pos is large-negative.

**Raw parametric peak:** **L27_H6** (`par_pos` +0.027). Llama analog: L1_H4.

### 7.2 FP16 Cell-3 (enlarge, α+=1 α−=0)

Source: `juice_stage2_gemma2_base_cell3/gemma2_base_cell3_summary.json` + bootstrap. Baseline **48%**, P/C/O **22/24/4**, coh 100%. (Llama base baseline was 16%.)


| Condition              | Heads                         | fc  | Δ [95% CI]            | excl. 0? | P/C/O   |
| ---------------------- | ----------------------------- | --- | --------------------- | -------- | ------- |
| baseline               | —                             | 48% | —                     | —        | 22/24/4 |
| ctx K10 enlarge        | directional ctx top-10        | 46% | **−2 [−8, +4]**       | no       | 23/23/4 |
| par K2 clean           | L31_H7, L39_H6                | 46% | **−2 [−8, +4]**       | no       | 23/23/4 |
| **par K1 raw**         | **L27_H6**                    | 36% | **−12 [−22, −4]**     | **yes**  | 28/18/4 |
| random-K10 enlarge     | seed 42, exclude ID union     | 42% | −6 [−14, +2]          | no       | 27/21/2 |


Random-K10 heads (locked for later suppress controls): L41_H11, L7_H3, L1_H9, L17_H13, L15_H13, L14_H7, L9_H1, L6_H9, L35_H9, L5_H10.

L27_H6: 6 context→parametric, `other` unchanged at 4. This is the only detected FP16 enlarge cell. Context K10 enlarge is a **null at a 48% ceiling**, not a Llama-style +10. Directional par K2 does not steer — as ID predicted (suppressors, not memory heads).

### 7.3 INT4 transfer of L27_H6 (the parametric object)

Source: `result-task-1/juice_stage2_gemma2_int4_transfer_l27h6/` (also `results (11)/`). Identify on FP16, apply on N_CALIB=128 INT4. Same 50 items, α+=1 α−=0. Smoke OK. Baseline **46%**, P/C/O **23/23/4**.


| Condition        | Heads      | INT4 fc | INT4 Δ [95% CI]     | excl. 0? | Sign vs FP16 | P/C/O   |
| ---------------- | ---------- | ------- | ------------------- | -------- | ------------ | ------- |
| baseline         | —          | 46%     | —                   | —        | —            | 23/23/4 |
| **L27_H6 K=1**   | L27_H6     | 38%     | **−8 [−16, −2]**    | **yes**  | same (−12)   | 27/19/4 |
| random-K1        | L40_H15    | 46%     | **0 [0, 0]**        | no       | —            | 23/23/4 |


L27_H6: 4 context→parametric, 0 reverse; `other` stays 4. Random-K1: **1000/1000 draws exactly zero** (no label flips). Head-specific transfer of a parametric object — the opposite of Llama L1_H4 on INT4.

### 7.4 FP16 context K10 suppress (ceiling check)

Enlarge of ctx K10 was a null. Suppress tests whether those heads have causal power in the other sign. Source: `task-2-res/juice_stage2_gemma2_fp16_context_suppress/`. α+=0 α−=1, enlarge empty. Same 50 items. Baseline 48% matches Cell-3 (22/24/4).

**Identified H−:** **−12 [−24, −2]** excl. 0. P/C/O **32/18/0**. 7 context→parametric; `other` 4→0 (3 other→parametric, 1 other→context). Coherence 100%.

Enlarge was a **ceiling**, not “no causal power.” Suppressing the directional context set moves the model toward parametric.

### 7.5 INT4 transfer of context K10 suppress

Source: `gemma-int4-kicu/juice_stage2_gemma2_int4_context_suppress/`. Same recipe, same locked random-K10 as Cell-3 (applied as H−, not resampled). Baseline 46% (23/23/4).


| Condition              | INT4 Δ [95% CI]      | excl. 0? | P/C/O   | notes |
| ---------------------- | -------------------- | -------- | ------- | ----- |
| identified H−          | **−8 [−16, −2]**     | **yes**  | 29/19/2 | same sign as FP16 −12; `other` not inflated (4→2); 4 context→parametric |
| **same random-K10 H−** | **−12 [−24, −2]**    | **yes**  | 29/17/4 | not head-specific |


Unlike L27_H6, INT4 suppress of the context set is **not** head-specific: the locked random-K10 also excludes 0.

### 7.6 FP16 random-K10 suppress (the missing control)

Source: `res-fp16-test-gemma/juice_stage2_gemma2_fp16_random_k10_suppress/`. Same locked 10 heads as the INT4 H− control, applied as H− on FP16.

**Random H−:** **−6 [−16, +4]** includes 0. P/C/O **25/21/4**. `other` unchanged. Comparing cell CIs, FP16 identified suppress (−12 excl. 0) *looked* specific vs this control (−6 incl. 0). That is a control-fails / treatment-detects reading, not a proven difference-of-deltas. §7.7 is the paired contrast.

### 7.7 Suppress specificity DiD (do not infer from overlapping cell CIs)

Source: `juice_stage2_gemma2_suppress_did/gemma2_suppress_specificity_did.json`. Script: `scripts/gemma2_suppress_specificity_did.py`. Each of 1000 draws resamples the same 50 eval IDs. Specificity = Δ_identified − Δ_random. DiD = spec_INT4 − spec_FP16. FP16 identified and FP16 random are **separate sessions** (own baselines); INT4 identified and random share one baseline. Percentile 95% CI.


| Estimand                         | Point | 95% CI         | excl. 0? | lt0 / eq0 / gt0 |
| -------------------------------- | ----- | -------------- | -------- | --------------- |
| Δ FP16 identified H−             | −12   | [−24, −2]      | yes      | 986 / 9 / 5     |
| Δ FP16 random H−                 | −6    | [−16, +4]      | no       | 837 / 75 / 88   |
| Δ INT4 identified H−             | −8    | [−16, −2]      | yes      | 983 / 17 / 0    |
| Δ INT4 random H−                 | −12   | [−24, −2]      | yes      | 987 / 9 / 4     |
| **specificity FP16**             | **−6** | **[−18, +6]** | **no**   | 801 / 83 / 116  |
| **specificity INT4**             | **+4** | **[−8, +16]** | **no**   | 196 / 99 / 705  |
| **DiD (INT4 spec − FP16 spec)**  | **+10** | **[−4, +24]** | **no**   | 59 / 29 / 912   |


Identified vs random is **not** a detected contrast on either precision. GPTQ changing that contrast is **not** detected at 95%. Do not claim “FP16 suppress is head-specific, INT4 is not” from overlapping cell CIs.

### 7.8 Cross-family sentence

Llama: context K10 **enlarge** transfers head-specifically; parametric does not (clean K2 null at α=1; L1_H4 wrong-sign on INT4).

Gemma: **L27_H6 raw par peak** transfers head-specifically (−12 FP16 / −8 INT4; random-K1 = 0). Directional par K2 does not steer. Ctx K10 enlarge is a null (48% ceiling). Ctx suppress is a detected FP16 −12; INT4 suppress of that set is not a detected identified-vs-random contrast; paired specificity DiD includes 0.

Quantization-selectivity is family-specific in *which* object survives, not a universal “context lives / parametric dies” rule.

---


## Number sources


| Claim                                          | File                                                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Stage 1 MR★ / AME / transitions                | `writeup/mini_pilot.md` (AME not re-fit this session)                                                  |
| Instruct vs base ID                            | `juice_stage2_base_id/base_id_comparison_summary.json`                                                 |
| FP16 base Cell-3                               | `juice_stage2_base_directional_cell3/base_directional_cell3_summary.json`                              |
| INT4 transfer + CIs (FP16 and N_CALIB=64 INT4) | `int_4_result/int4_transfer_summary.json`, `int_4_result/int4_transfer_bootstrap.json`                 |
| N_CALIB=128 transfer + CIs                     | `this_is_the_128/int4_ncalib128_transfer_bootstrap.json`, `this_is_the_128/int4_transfer_summary.json` |
| N64 vs N128 comparison table                   | `this_is_the_128/n64_vs_n128_comparison.json`                                                          |
| Native INT4 ID                                 | `here_is_the/int4_native_id_summary.json`                                                              |
| Native INT4 clean K2 apply (L16_H30, L17_H27)  | `ungabunga_result/juice_stage2_int4_ncalib128/int4_native_clean_k2_summary.json`, `int4_native_clean_k2_bootstrap.json` |
| Reverse transfer + native INT4 context K10     | `unga-bunga-23/juice_stage2_int4_ncalib128/int4_reverse_and_native_k10_summary.json`, `int4_reverse_and_native_k10_bootstrap.json` |
| Cell A text / par_pos diagnostics              | `unga-bunga-23/cell_a_diagnostics.json`, `unga-bunga-23/cell_a_inspection.md` |
| Cell A α-sweep (INT4 K10 → FP16)               | `resultofthe/juice_stage2_int4_ncalib128/cell_a_alpha_sweep_summary.json`, `cell_a_alpha_sweep_bootstrap.json` |
| Cell B text inspection                         | `unga-bunga-23/cell_b_inspection.md`, `unga-bunga-23/cell_b_diagnostics.json` |
| Jaccard bootstrap                              | `here_is_the/jaccard_bootstrap.json`                                                                   |
| Random-K (N=64)                                | `here_is_the/int4_random_k_summary.json`                                                               |
| Random-K (N=128)                               | `this_is_the_128/int4_random_k_summary.json`                                                           |
| L1_H4 / α-grid / suppress                      | `here_is_the-2/int4_parametric_fixes_summary.json`                                                     |
| dir_par top-20                                 | `juice_stage2_int4_native_id/directional_par_top20.json`                                               |
| FP16 per-head scores                           | `juice_stage2_base_id/id_diagnostics_raw.json`, `directional_rerank_base.json`                         |
| Gemma FP16 ID                                  | `juice_stage2_gemma2_base_id/base_id_comparison_summary.json`                                          |
| Gemma FP16 Cell-3                              | `juice_stage2_gemma2_base_cell3/gemma2_base_cell3_summary.json`, `gemma2_base_cell3_bootstrap.json`     |
| Gemma INT4 L27_H6 transfer                     | `result-task-1/juice_stage2_gemma2_int4_transfer_l27h6/` (dup `results (11)/`)                          |
| Gemma FP16 ctx K10 suppress                    | `task-2-res/juice_stage2_gemma2_fp16_context_suppress/`                                                |
| Gemma INT4 ctx K10 suppress                    | `gemma-int4-kicu/juice_stage2_gemma2_int4_context_suppress/`                                            |
| Gemma FP16 random-K10 suppress                 | `res-fp16-test-gemma/juice_stage2_gemma2_fp16_random_k10_suppress/`                                     |
| Gemma suppress specificity DiD                 | `juice_stage2_gemma2_suppress_did/gemma2_suppress_specificity_did.json`                                |




### Flags (do not guess past these)

1. Cluster-robust AME **+10.3 [+3.1, +17.5]** is from the Stage 1 memo, not a JSON in-repo. If the paper needs a methods appendix table, re-run the mixed model from `detail_relabeled_scorer_fix.jsonl`.
2. Native INT4 clean pair **L16_H30, L17_H27 has been steered** on the N_CALIB=128 checkpoint: **−2 [−8, +4]**, includes 0. Source: `ungabunga_result/`. Identified on the N=64 native-ID run; applied on N=128. Not a detected effect.
3. Stage 2 headline INT4 is **N_CALIB=128** (`llama31_8b_base_gptq_int4_ncalib128`). N_CALIB=64 is retained as a robustness check; no “excludes 0” verdict flipped on the three re-run cells. Native ID / α-grid / L1_H4 / H− were not repeated at N=128. Native *apply* of the clean pair was run at N=128. Reverse transfer (Cell A) is a detected +48 that is **not** the same circuit (manual inspection + α-sweep). Native INT4 context K10 (Cell B) is confirmed: +18, exam-meta 0/19, matches forward transfer.
4. Jaccard noise uses **25** draws, not 20+ as the brief’s ideal.
5. Instruct Cell-3 parametric 0/−2 pp has **no** paired bootstrap in-repo (point estimates only).
6. CIs are reported in whole percentage points to match a 2 pp (1/50) grid; JSON stores them as floats (e.g. 3.999… → +4).
7. Cell A’s scorer-level +48 was investigated and **rejected** as reverse transfer of FP16’s K10 (exam-meta 13/32 of context labels at α=1; α-sweep floor +34 pp with 58% exam-meta at α=0.25). Methods lesson, same as the 1B brevity confound and the α=3/5 trap: scorer-level detection is necessary but not sufficient — read the completions before claiming a mechanistic result. Do not silently drop the +48; report it as a diagnostic catch.
8. Gemma Stage 2 is **base only**. No Gemma Instruct MR★, no native INT4 re-ID, no reverse transfer. Do not copy Llama cells into those blanks.
9. Gemma suppress “head-specific on FP16, not on INT4” is a **cell-CI** reading. The paired specificity CIs both include 0; DiD **+10 [−4, +24]** includes 0. Publish the DiD, not the overlapping-CI story.
10. Never requantize Gemma. Canonical INT4 is `gemma2_9b_base_gptq_int4_ncalib128`. Do not use `gemma2_gptq_ncalib8`.

