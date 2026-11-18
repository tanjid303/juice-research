# Mini-pilot decision memo (§5.1)

**Status:** decided (both gates). **1B → pivot (b)** (null / confound screen). **8B → go** — double-filtered MR★ moves **−23.8 pp** (CI excludes 0), well above the ≳8–10 pp bar; proceed to Stage 2 JuICE on 8B (not on 1B items).

This memo is the Week-1 / Stage-1 behavioral gate. Numbers below for **8B use the corrected scorer** (`rejects_context` for denied context mentions; bare entity names count as context). Buggy pre-fix AME (+20.6 pp) is superseded.

---

# Part A — Llama-3.2-1B-Instruct (prior gate; pivot b)

## Run metadata (1B)

| Field | Value |
|-------|--------|
| Date | 2026-08-24 |
| Machine | RTX 3050 Ti laptop (4 GB) |
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Precisions | FP16 vs GPTQ-INT4 (`fbaldassarri/...-gptq-int4-gs128-sym`) |
| Data | `data/conflictqa_pilot250_popular.jsonl` |
| n / shared double-filter | 250 / **115** |
| Artifacts | `mini_pilot/outputs/summary_20260824T130134Z.csv`, `detail_…`, `stats_…` |

## Results (1B)

| Precision | Closed-book | MR (unfiltered) | MR★ (shared 115) |
|-----------|-------------|-----------------|------------------|
| FP16 | 143/250 | 58/250 (23.2%) | 50/115 (43.5%) |
| INT4 | 127/250 | 60/250 (24.0%) | 52/115 (45.2%) |

**Δ MR★ = +1.74 pp**, bootstrap CI **[−5.22, +9.57]** — includes 0.

Labels on shared 115: parametric **flat** (50→52); context **drops** (32→19); other **rises** (33→44). Mixed AME on `follows_context` ≈ **−9 to −11 pp** was a **brevity / under-commitment confound** (INT4 conflict outputs mean 2.27 vs 5.47 tokens), not source arbitration.

**1B decision:** pivot (b) — negative dissociation / confound screen; jump to 8B. Do not claim a 1B arbitration effect; do not start JuICE on these 1B items.

---

# Part B — Llama-3.1-8B-Instruct (serving-scale gate; go)

## Run metadata (8B)

| Field | Value |
|-------|--------|
| Date | 2026-08-24 (Kaggle run); scorer relabel 2026-08-25 |
| Machine | Kaggle 2× Tesla T4 (16 GB each) |
| Model FP16 | `meta-llama/Llama-3.1-8B-Instruct` (`device_map="auto"`) |
| Model INT4 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` (single GPU `{'': 0}`; **gptqmodel** `BACKEND.TORCH` — Marlin JIT failed on sm_75) |
| Decode | greedy, T=0, batch=1 |
| Data | same `conflictqa_pilot250_popular.jsonl` (250 items) |
| Shared double-filter n | **164** (FP16 ∩ INT4 closed-book correct) |
| Bootstrap | seed 42; 1000 resamples of the 164-item set |
| Inference artifacts | `Downloads/mini_pilot_8b_outputs/detail_20260824T180723Z.jsonl`, `summary_…`, `stats_…` (stats AME is **pre-fix**) |
| Corrected labels / stats | `detail_relabeled_scorer_fix.jsonl`, `scorer_fix_report.txt` |
| Scorer | `common/scoring.py` — labels `{parametric, context, rejects_context, other}` |

**INT4 VRAM note:** peak ~19.6 GiB combined reported with residual FP16 tensors on GPU1; INT4 weights fit one T4 as intended.

---

## Results (headline table — corrected labels)

| Precision | Closed-book | MR (unfiltered) | MR★ (shared 164) |
|-----------|-------------|-----------------|------------------|
| FP16 | **182/250** (72.8%) | **92/250** (36.8%) | **76/164** (46.3%) |
| INT4 | **175/250** (70.0%) | **48/250** (19.2%) | **37/164** (22.6%) |

**Δ MR★ (INT4 − FP16) = −39/164 = −23.8 pp.**

Bootstrap 95% CI: **[−31.7, −17.1] pp. Excludes 0.**

Closed-book is nearly stable (−7 items). Unfiltered MR also drops hard (−17.6 pp), so this is **not** “pure forgetting that vanishes after the double filter.” After the filter, the arbitration estimand still moves by **≃24 pp** — clear go on the §5.1 MR★ clause.

### Secondary: mixed AME on `follows_context` (corrected scorer)

`follows_context := conflict_label == "context"` (excludes `rejects_context`). Same 164 items.

| Estimand | Point | 95% CI | excludes 0? |
|----------|-------|--------|-------------|
| Mixed VB AME | **+16.1 pp** | [+5.9, +26.2] | yes |
| Item-averaged AME | +9.8 pp | — | — |
| Cluster-robust logit AME | **+10.3 pp** | [+3.1, +17.5] | yes |
| Raw Δ P(context) | **+10.4 pp** | (52→69)/164 | — |

**Pre-fix inflated AME was +20.6 pp** (cluster-robust +12.7; raw +12.8). After stripping rejection→context mislabels, report **~+10 pp** (cluster / raw) as the honest context-following shift; mixed VB (+16.1) remains directionally consistent but larger.

---

## Labels on the shared 164 (conflict condition, corrected)

| Label | FP16 | INT4 | Δ |
|-------|------|------|---|
| parametric | 76 | 37 | **−39** |
| context | 52 | 69 | **+17** |
| rejects_context | 13 | 22 | +9 |
| other | 23 | 36 | +13 |

Unlike 1B: parametric **collapses**; context **rises**; other also rises. The MR★ move is parametric loss, not a flat parametric with context→other reshuffle.

### Transition table FP16 → INT4 (shared 164)

| | → parametric | → context | → rejects_context | → other |
|---|---:|---:|---:|---:|
| **parametric →** | 33 | **20** | 10 | 13 |
| **context →** | 3 | 41 | 5 | 3 |
| **rejects_context →** | 0 | 3 | 5 | 5 |
| **other →** | 1 | 5 | 2 | 15 |

Dominant mass off parametric: **20 → context**, **10 → rejects_context**, **13 → other**.

---

## Diagnostics (same checks as 1B — plus scorer repair)

### 1. Scorer bug and fix (required before trusting AME)

~10/26 of the pre-fix `parametric→context` flips were **parametric rejections that named the context entity** (e.g. `Sitten is not the capital of Lebanon`) and were scored `context` by substring match.

**Fix:** negation / confirmation-denial frames → `rejects_context` unless the parametric answer is clearly asserted; bare single-entity answers matching the context gold → `context` (fixed Kareličy / Hamilton / Iconium, which had been stuck in `other`).

Relabel only — no re-inference. Spot-checks: all ten false-context rejections → `rejects_context`; three bare names → `context`.

### 2. Brevity / under-commitment (Llama-3.1 tokenizer, no special tokens)

| Subset | n | FP16 mean (med) | INT4 mean (med) | Δ mean |
|--------|---:|-----------------|-----------------|-------:|
| parametric → context | 26* | 14.4 (7.5) | 10.5 (7.0) | −4.0 |
| parametric → other | 17* | 14.7 (11) | 8.7 (7) | −6.1 |
| parametric → parametric (stable) | 33* | 8.7 (8) | 9.4 (8) | +0.6 |
| All shared conflict outs | 164 | 10.5 (7) | 9.0 (7) | ~−1.5 |

\*pre-fix flip sets used for the length diagnostic; pattern is unchanged in spirit after relabel.

**vs 1B:** medians on shared set are **matched** (7 vs 7); stable-parametric lengths do not shrink. Not the 1B “INT4 mean 2.3 tok” collapse. Remaining risk is **bimodal** on some p→c items: FP16 long correction (hit 32-tok cap) → INT4 bare context name (2–5 tok). That is slot-filling / under-specification for a subset, not a global brevity confound.

### 3. Manual read of parametric→context (pre-fix bucket; still informative)

Two coherent subtypes after scorer fix:

- **Genuine context adoption** — INT4 answers with the counterfactual entity (e.g. `Karl Mak Ka`, `Dean White`, `Brades.`, `Vilne.`).
- **Rejection / hedge** — now `rejects_context` (e.g. `Tokyo is not the capital of Kangra`, `Sitten is not…`).

`parametric→other` at 8B is **not** mostly 1B-style 1-token stubs: mix of truncated refusals (`I can't verify that information.`), wrong entities, and bare context names that the old scorer missed (now fixed to `context`).

---

## Go / pivot / debug (against §5.1, written)

§5.1 go if: double-filtered MR moves ≳8–10 pp, **or** closed-book is stable but conflict behavior is not.

| Clause | 1B | **8B (corrected)** |
|--------|----|--------------------|
| MR★ ≳8–10 pp | No (+1.7; CI ∋ 0) | **Yes (−23.8; CI [−31.7, −17.1])** |
| Pivot: MR vanishes after double filter | Unfiltered already null | **No** — unfiltered MR also drops; MR★ still −24 pp |
| CB stable, conflict not | CB unstable; conflict = brevity/`other` | CB nearly stable (−2.8 pp); conflict = parametric↓ + context↑ |

**Decision: go to Stage 2 on 8B.**

Reasons:

1. **RQ0 at serving scale fires on the primary estimand.** INT4 reduces double-filtered parametric answers under conflict by ~24 pp with a CI that excludes 0 — the opposite of the 1B null.
2. **This is not the 1B confound.** Parametric falls; context rises; brevity is not systematic; the AME survives scorer repair at ~+10 pp (cluster/raw).
3. **Brief kill/pivot table:** “Effect present in 8B on MR★ → proceed to Stage 2.” That clause is met. The 1B-only path remains forbidden.
4. **Caveats that do not block go:** (i) report AME on corrected labels; (ii) keep an eye on bare-name / short-slot answers in any JuICE item selection; (iii) INT4 path used Torch GPTQ backend on T4 — prefer a clean Marlin/A4500 GPTQ load for Stage 2 cells if available.

## Checkpoint QC note (8B INT4)

Checkpoint: `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`

- **Config / provenance:** vanilla GPTQ, bits=4, group_size=128, sym — matches repo claim
- **g_idx:** properly populated, no repair needed
- **Coherence:** 10/10 factual prompts generated coherently
- **Perplexity:** sample-dependent, **+2.5% to +27.4%** degradation across 4 text samples
  - physics / history prose: **+2.5–4.8%**, healthy GPTQ range
  - Amazon-rainforest / meta-LLM prose: **+17.0–27.4%**, elevated
  - The one exact-text-matched comparison against the 1B checkpoint (same Amazon passage) showed **+27.4%** degradation at 8B vs **+2.6%** at 1B — largest and least explained data point
  - “Low FP16 baseline inflates relative %” was considered and does **not** fully hold: sample B has a nearly identical baseline PPL to sample A but shows only **+4.8%** degradation, with a ~5× smaller absolute PPL increase
  - Mechanism for the sample-A/D outliers is **unresolved**, not confirmed artifact

**Decision:** proceeded to Stage 2 on this checkpoint. Coherence + structural QC (config / g_idx) were treated as sufficient gating evidence; PPL was informative but not decisive, given the inconsistent spread. MR★ / AME behavioral results do not depend on PPL being pristine.

**Follow-up before paper writeup:** self-quantize GPTQ (own calibration data) and re-run this same matched-sample PPL check as the checkpoint used in the final paper’s Stage 1/2 numbers — closes the open question rather than relying on a third-party checkpoint’s unexplained perplexity variance.

### What we will do next (and will not)

**Do:**

- Stage 2 JuICE on **Llama-3.1-8B** (instruct headline; base confound check per Fix 3), FP16 reproduction gate first.
- Lock corrected scorer into all future ConflictQA labeling; keep `rejects_context` in breakdowns (AME still uses `== "context"` only).
- Update `PRE_REG.md` estimands to cite corrected 8B MR★ / AME as the Stage-1 behavioral finding that authorizes Stage 2.
- Prefer A4500 (or equivalent) for Stage 2 batch work; Kaggle T4×2 was sufficient for this behavioral cell.
- Before paper writeup: self-quantize GPTQ with own calibration and re-run the matched-sample PPL check (see Checkpoint QC note above).

**Do not:**

- Treat the pre-fix +20.6 AME as the publishable number.
- Start Stage 2 identify/apply/Jaccard on **1B** ConflictQA items.
- Claim the 1B AME as arbitration evidence.
- Skip JuICE FP16 reproduction on 8B-base (Fix 2 gate).

### One-sentence locked takeaways

**1B:** On Llama-3.2-1B-Instruct, GPTQ-INT4 does not shift double-filtered memorization under conflict (MR★ +1.7 pp, CI includes 0); the apparent context-following drop was a brevity/`other` confound.

**8B:** On Llama-3.1-8B-Instruct, GPTQ-INT4 **does** shift arbitration: MR★ **−23.8 pp** [−31.7, −17.1]; after scorer repair, `follows_context` rises by ~**+10 pp** (cluster-robust / raw). Per §5.1, **go to Stage 2 on 8B**.
