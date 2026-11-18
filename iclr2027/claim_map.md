# Frozen claim map (pre-restructure)

Snapshot of headline/supporting numbers for Findings 1–3. Values are the **question-aware** estimand unless marked original-matcher. All CIs are percentile 95% from 1000 paired bootstrap draws, seed 42. Line ranges refer to `iclr2027/main.tex` (identical in `iclr2027/main2.tex` at copy time). Do not invent cells marked NOT RUN / MISMATCHED / NOT EXPANDED.

Convention: Δ = steered minus baseline `follows_context` (pp). Gold-PAR-only Δ counts a leave-context move as success only if the steered text mentions the gold parametric entity.

---

## Finding 1 — Llama context K10 (wrap artifact)

Locked FP16 directional context K10 (enlarge, α+=1, α−=0): **L4_H9, L9_H0, L1_H11, L4_H5, L19_H27, L17_H12, L7_H26, L14_H23, L3_H22, L31_H20**. Source: `llama1/juice_stage2_llama_base_n100/llama_base_n100_summary.json`. Same locked set on INT4 N_CALIB=128.

Random context K10 (seed 42): **L20_H23, L3_H19, L0_H25, L24_H0, L8_H30, L7_H31, L7_H8, L4_H18, L23_H27, L3_H9**. Same summary file.

| ID | Claim | Value + CI | Heads | Precision | n | N_CALIB | Source file | main.tex lines |
|---|---|---|---|---|---|---|---|---|
| F1-1 | Llama ctx K10 original-matcher | **+8 [+2, +14]** detected | locked K10 above | FP16 | 100 | — | `llama1/juice_stage2_llama_base_n100/steer_context_k10.json`; fmt in `stage2_question_aware_rescore/headline_first.json` id `llama_n100_fp16_ctx_k10` | L34 (abstract); L70; L227; L602 |
| F1-2 | Llama ctx K10 QA (inclusive) | **+4 [+0, +8]** null | locked K10 | FP16 | 100 | — | same; `headline_first.json` rescored | L70; L227; L248; L601 |
| F1-3 | Llama ctx K10 original-matcher transfer | **+15 [+8, +23]** detected | locked K10 | INT4 | 100 | 128 | `llama2/juice_stage2_int4_ncalib128_n100/steer_context_k10.json`; `headline_first.json` id `llama_n100_int4_ctx_k10` | L34; L70; L227; L237; L600 |
| F1-4 | Llama ctx K10 QA transfer (inclusive) | **+1 [+0, +3]** null | locked K10 | INT4 | 100 | 128 | same; rescored | L34; L70; L227; L248; L600 |
| F1-5 | Wrap count on original INT4 flips | **17/17** original INT4 fc-flips are same-entity wraps (QA leftover: 1 gold-CTX switch) | locked K10 | INT4 | 100 | 128 | `stage2_question_aware_rescore/canvas_entity_rows.json` id `llama_n100_int4_ctx_k10` (`owrap`: 17) | L34; L227; L600 |
| F1-6 | Random ctx K10 QA | **+0 [−4, +4]** null | random K10 above | FP16 | 100 | — | `llama1/juice_stage2_llama_base_n100/random_context_k10.json` | L70; L227; L249 |
| F1-7 | Random ctx K10 QA | **−1 [−4, +2]** null | random K10 | INT4 | 100 | 128 | `llama2/juice_stage2_int4_ncalib128_n100/random_context_k10.json` | L70; L227; L249 |

Finding 1 verdict to freeze: original-matcher transfer is a wrap; QA cells are null vs random.

---

## Finding 2 — Llama clean K2 (output-level gold-PAR transfer)

Locked pair: **L20_H14 + L26_H15**. α+=1, α−=0. Headline INT4 checkpoint is N_CALIB=128.

Gold-PAR and inclusive come from `stage2_question_aware_rescore/k2_gold_par_only.json`. Inclusive QA Δ also matches `headline_first.json` rescored.

| ID | Claim | Value + CI | Heads | Precision | n | N_CALIB | Source file | main.tex lines |
|---|---|---|---|---|---|---|---|---|
| F2-1 | Gold-PAR-only | **−8 [−14, −3]** detected (9 gold-PAR / 7 leave; boot hi −2.975, reported as −3) | L20_H14+L26_H15 | FP16 | 100 | — | `k2_gold_par_only.json` `llama_n100_fp16_par_k2`; steered `llama1/.../steer_parametric_k2_clean.json` | L34; L70; L135; L159; L262; L274; L432; L603 |
| F2-2 | Inclusive QA | **−15 [−22, −7]** detected | L20_H14+L26_H15 | FP16 | 100 | — | same; `headline_first.json` rescored | L34; L135; L262; L274 |
| F2-3 | Original-matcher (inclusive, not gold-PAR) | **−8 [−14, −3]** detected | L20_H14+L26_H15 | FP16 | 100 | — | `headline_first.json` original | L603 (caption distinguishes this from gold-PAR) |
| F2-4 | Gold-PAR-only transfer | **−7 [−12, −2]** detected (7 gold-PAR / 5 leave) | L20_H14+L26_H15 | INT4 | 100 | 128 | `k2_gold_par_only.json` `llama_n100_int4_par_k2`; `llama2/.../steer_parametric_k2_clean.json` | L34; L70; L262; L275; L432; L603 |
| F2-5 | Inclusive QA transfer | **−12 [−19, −6]** detected | L20_H14+L26_H15 | INT4 | 100 | 128 | same; `headline_first.json` rescored | L34; L262; L275; L435 |
| F2-6 | Original-matcher transfer | **−4 [−8, −1]** detected | L20_H14+L26_H15 | INT4 | 100 | 128 | `headline_first.json` original | L603 |
| F2-7 | Gold-PAR-only | **−6 [−14, +0]** null | L20_H14+L26_H15 | FP16 | 50 | — | `k2_gold_par_only.json` `llama_n50_fp16_par_k2c`; `juice_stage2_base_directional_cell3/steer_parametric_k2_clean.json` | L262; L274 |
| F2-8 | Inclusive QA | **−14 [−24, −4]** detected | L20_H14+L26_H15 | FP16 | 50 | — | same (`inclusive`) | not a main-table column; keep as supporting |
| F2-9 | Gold-PAR-only (headline n=50 INT4) | **−8 [−16, −2]** detected | L20_H14+L26_H15 | INT4 | 50 | **128** | `k2_gold_par_only.json` `llama_n50_int4_n128_par_k2c`; `this_is_the_128/steer_parametric_k2_clean.json` | L262; L275 |
| F2-10 | Inclusive QA | **−14 [−26, −4]** detected | L20_H14+L26_H15 | INT4 | 50 | 128 | same | not a main-table column |
| F2-11 | Original-matcher n=50 N128 (distinct from F2-6) | **−4 [−10, +0]** null | L20_H14+L26_H15 | INT4 | 50 | 128 | same `original` | L435; L592 caption |

**FLAG — Llama random-K2: EXISTS BUT MISMATCHED.** Not a valid control for F2-1–F2-6.

| ID | Claim | Value + CI | Heads | Precision | n | N_CALIB | Source file | main.tex lines |
|---|---|---|---|---|---|---|---|---|
| GAP-K2 | Random parametric K2 | orig/QA **+4 [+0, +10]** null | **L13_H2 + L16_H20** (seed 44), **not** L20_H14+L26_H15 | INT4 | **50** | **64** | `here_is_the/int4_random_k_summary.json`; `here_is_the/random_parametric_k2.json` | L262 (disclosure already present) |

No random-K2 file exists at n=100, N128, or FP16.

---

## Finding 3 — Gemma identified H− (selective / attenuated)

Identified H− = directional **context K10** applied as suppress (α+=0, α−=1, enlarge empty): **L40_H14, L17_H7, L24_H15, L22_H0, L32_H11, L25_H0, L3_H9, L8_H9, L7_H6, L29_H5**. Source: `last-one-g/gemma_n100_fp16/gemma2_base_cell3_summary.json` and `iclr2027/main.tex` L514.

Locked random H− (same seed-42 random K10, applied as suppress): **L41_H11, L7_H3, L1_H9, L17_H13, L15_H13, L14_H7, L9_H1, L6_H9, L35_H9, L5_H10**. Source: `last-one-g/gemma_n100_fp16_random_hminus/random_k10_heads.json`.

Inclusive QA from `master_comparison.json`. Gold-PAR from `stage2_question_aware_rescore/hminus_gold_par_only.json`. Specificity/DiD from `master_comparison.json` derived cells (n=100) and `last-one-g/gemma_hminus_question_aware_rescore.json` (n=50).

| ID | Claim | Value + CI | Heads | Precision | n | Source file | main.tex lines |
|---|---|---|---|---|---|---|---|
| F3-1 | Identified H− inclusive | **−19 [−27, −11]** detected | identified K10 as H− | FP16 | 100 | `last-one-g/gemma_n100_fp16_hminus/suppress_context_k10.json` | L34; L71; L157; L326; L498; L521; L605 |
| F3-2 | Identified H− gold-PAR | **−15 [−23, −8]** detected (16 gold-PAR / 4 leave) | same | FP16 | 100 | `hminus_gold_par_only.json` `n100_fp16` | L34; L157; L326; L498; L521; L605 |
| F3-3 | Random H− inclusive | **−1 [−5, +2]** null | random K10 as H− | FP16 | 100 | `last-one-g/gemma_n100_fp16_random_hminus/random_suppress_k10.json` | L34; L71; L326; L500; L521; L605 |
| F3-4 | Specificity (id − random) | **−18 [−27, −10]** detected | derived | FP16 | 100 | `master_comparison.json` id `specificity FP16 (id − random)` rescored | L326; L502 |
| F3-5 | Identified H− inclusive | **−11 [−18, −4]** detected | identified K10 as H− | INT4 N128 | 100 | `last-one-g/gemma_n100_int4_hminus/suppress_context_k10.json` | L34; L71; L157; L326; L499; L521; L605 |
| F3-6 | Identified H− gold-PAR | **−10 [−17, −4]** detected (11 gold-PAR / 1 leave) | same | INT4 N128 | 100 | `hminus_gold_par_only.json` `n100_int4` | L34; L157; L326; L499; L521; L605 |
| F3-7 | Random H− inclusive | **−1 [−5, +3]** null | random K10 as H− | INT4 N128 | 100 | `last-one-g/gemma_n100_int4_hminus/random_suppress_k10.json` | L34; L71; L326; L501; L521; L605 |
| F3-8 | Specificity | **−10 [−18, −2]** detected | derived | INT4 N128 | 100 | `master_comparison.json` id `specificity INT4 (id − random)` rescored | L326; L503 |
| F3-9 | DiD (spec INT4 − spec FP16) | **+8 [+1, +16]** detected (FP16 specificity larger → attenuation on INT4) | derived | both | 100 | `master_comparison.json` id `DiD (spec INT4 − spec FP16)` rescored | L326; L504 |
| F3-10 | Identified H− inclusive | **−14 [−26, −4]** detected | identified K10 as H− | FP16 | 50 | `task-2-res/juice_stage2_gemma2_fp16_context_suppress/suppress_context_k10.json` | L498 |
| F3-11 | Identified H− gold-PAR | **−10 [−20, +0]** null | same | FP16 | 50 | `hminus_gold_par_only.json` `n50_fp16` | L326 (“n=50 conservative count includes 0”) |
| F3-12 | Random H− inclusive | **−4 [−10, +0]** null | random K10 as H− | FP16 | 50 | `res-fp16-test-gemma/.../random_suppress_k10.json` | L500 |
| F3-13 | Specificity | **−10 [−22, +2]** null | derived | FP16 | 50 | `gemma_hminus_question_aware_rescore.json` n=50 derived | L502; L363 |
| F3-14 | Identified H− inclusive | **−8 [−16, +0]** null | identified K10 as H− | INT4 N128 | 50 | `gemma-int4-kicu/.../suppress_context_k10.json` | L499 |
| F3-15 | Identified H− gold-PAR | **−8 [−16, +0]** null | same | INT4 N128 | 50 | `hminus_gold_par_only.json` `n50_int4` | L326 |
| F3-16 | Random H− inclusive | **−2 [−10, +4]** null | random K10 as H− | INT4 N128 | 50 | `gemma-int4-kicu/.../random_suppress_k10.json` | L501 |
| F3-17 | Specificity | **−6 [−18, +6]** null | derived | INT4 N128 | 50 | `gemma_hminus_question_aware_rescore.json` n=50 derived | L503 |
| F3-18 | DiD | **+4 [−4, +14]** null | derived | both | 50 | same | L504 |

---

## Finding 3 supporting — Gemma context K10 enlarge

Same identified context K10 as H−, but **enlarge** (α+=1, α−=0). Random enlarge uses the same locked random K10 as above.

| ID | Claim | Value + CI | Heads | Precision | n | Source file | main.tex lines |
|---|---|---|---|---|---|---|---|
| E-1 | Ctx K10 enlarge QA | **+7 [+2, +12]** detected | identified ctx K10 | FP16 | 100 | `last-one-g/gemma_n100_fp16/steer_context_k10.json` | L34; L71; L324; L476; L518; L606 |
| E-2 | Random K10 enlarge QA | **+0 [+0, +0]** null | random K10 | FP16 | 100 | `last-one-g/gemma_n100_fp16/random_context_k10.json` | L324; L483; L518 |
| E-3 | Ctx K10 enlarge QA | **+5 [+1, +9]** detected | identified ctx K10 | INT4 N128 | 100 | `last-one-g/gemma_n100_int4_headline/steer_context_k10.json` | L34; L71; L324; L477; L518; L606 |
| E-4 | Random K10 enlarge QA | **+3 [+0, +7]** null (overlaps E-3) | random K10 | INT4 N128 | 100 | `last-one-g/gemma_n100_int4_headline/random_context_k10.json` | L324; L518 |
| E-5 | Ctx K10 enlarge QA | **+4 [+0, +10]** null | identified ctx K10 | FP16 | 50 | `juice_stage2_gemma2_base_cell3/steer_context_k10.json` | L476; L518 |
| E-6 | Random K10 enlarge QA | **+0 [+0, +0]** null | random K10 | FP16 | 50 | `juice_stage2_gemma2_base_cell3/random_context_k10.json` | L483 |
| E-7 | Ctx K10 enlarge INT4 n=50 | **NOT RUN** | — | INT4 | 50 | no steered file in rescore cell list | L477 shows `---` |

INT4 n=100 identified enlarge does **not** separate from random (E-3 vs E-4). Freeze that as a caveat, not as an identified-specific object.

---

## Finding 3 supporting — Gemma L27_H6 (wrap)

Raw parametric peak **L27_H6** (K=1 enlarge). Random K=1 control: **L40_H15**.

The four QA cells:

| ID | Claim | Value + CI | Heads | Precision | n | Source file | main.tex lines |
|---|---|---|---|---|---|---|---|
| L27-1 | QA | **+0 [+0, +0]** hard zero (1000/1000 draws exactly 0) | L27_H6 | FP16 | 50 | `juice_stage2_gemma2_base_cell3/steer_parametric_k1_raw.json` | L71; L324; L480; L515 |
| L27-2 | QA | **+0 [+0, +0]** hard zero | L27_H6 | INT4 | 50 | `result-task-1/juice_stage2_gemma2_int4_transfer_l27h6/steer_parametric_k1_raw.json` | L324; L481; L515 |
| L27-3 | QA | **+0 [+0, +0]** hard zero | L27_H6 | FP16 | 100 | `last-one-g/gemma_n100_fp16/steer_parametric_k1_raw.json` | L480 |
| L27-4 | QA | **+0 [−3, +3]** null (**not** a hard zero) | L27_H6 | INT4 N128 | 100 | `last-one-g/gemma_n100_int4_headline/steer_parametric_k1_raw.json` | L71; L324; L481; L515 |
| L27-orig-a | Original-matcher (pre-audit) | **−12 [−22, −4]** detected | L27_H6 | FP16 | 50 | `headline_first.json` original | L515; L604 |
| L27-orig-b | Original-matcher | **−8 [−16, −2]** detected | L27_H6 | INT4 | 50 | same | L515; L604 |

Freeze: “hard zero” applies to n=50 both precisions (and n=100 FP16). n=100 INT4 is a null CI that includes 0, not 1000/1000 zeros.

---

## Explicit gaps (do not fill)

### GAP A — Gemma clean-K2 INT4: NOT RUN

- FP16 n=100 QA **does** exist: **−6 [−12, −1]** detected (`last-one-g/gemma_n100_fp16/steer_parametric_k2_clean.json`; heads **L31_H7, L39_H6**; main.tex L324, L478).
- INT4 transfer of that locked pair: **no file**. Table row L479 already says `not run`.
- Do not treat Gemma’s parametric direction as tested-and-null on INT4.

### GAP B — Llama random-K2: EXISTS BUT MISMATCHED

- Existing cell: L13_H2+L16_H20, n=50, INT4 N_CALIB=64, seed 44, +4 [+0, +10].
- Headline cell: L20_H14+L26_H15, n=100, INT4 N_CALIB=128 (and FP16 n=100).
- Different heads, different n, different calibration size. Not a matched control.

### GAP C — Context K5: NOT EXPANDED to n=100

Existing n=50 only:

| Cell | QA Δ | Source | n | N_CALIB |
|---|---|---|---|---|
| FP16 ctx K5 | **+8 [+2, +16]** | `juice_stage2_base_directional_cell3/steer_context_k5.json` | 50 | — |
| INT4 ctx K5 | **+2 [+0, +8]** | `int_4_result/steer_context_k5 (1).json` | 50 | 64 |
| Random ctx K5 | orig **+2 [+0, +6]** | `here_is_the/random_context_k5.json` | 50 | 64 |

No K5 under `llama1/` or `llama2/` n=100 dirs. Methods gate no longer names K5; Appendix D L526 states the n=50-only fact.

---

## Draft disclosure sentences (not inserted yet)

**A. Gemma clean-K2 INT4 (NOT RUN)**

> Gemma clean-K2 enlarge (L31_H7, L39_H6) was run on FP16 at n=100 (question-aware Δ −6 [−12, −1]). The INT4 transfer of that locked pair was not run; we do not treat the missing cell as evidence that Gemma’s parametric direction fails to transfer.

**B. Llama random-K2 (EXISTS BUT MISMATCHED)**

> The only random parametric-K2 result uses L13_H2 and L16_H20 on the n=50, N_CALIB=64 INT4 checkpoint (+4 [+0, +10], null). It is not a matched control for the locked L20_H14+L26_H15 headline at n=100, N_CALIB=128; no matched random-K2 control was run for that cell.

**C. Context K5 (NOT EXPANDED)**

> Context K5 was run only at n=50 (FP16 and INT4 N_CALIB=64) and was not expanded to the n=100 question-aware audit; all Llama context headlines use K10.

**D. Gemma context K10 enlarge INT4 n=50 (NOT RUN)**

> Gemma context K10 enlarge was run on INT4 at n=100, and on FP16 at both n=100 and n=50. The INT4 transfer of this condition at n=50 was not run.

**E. Gemma L27_H6 FP16 random K1 (NOT RUN)**

> The original Gemma raw parametric peak (L27_H6) at n=50 was compared against an INT4 random K1 control (L40_H15, +0 [−6, +6], null). However, the corresponding FP16 random K1 control was not run at n=50.

These five sentences are for sign-off only. They are not yet written into `main2.tex`.
