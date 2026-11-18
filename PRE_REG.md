# Pre-registration stub (fill before 8B Stage 1)

> **Do not run Llama-3.1-8B experiments from this 3050 Ti setup.** This file is a placeholder per PROJECT_BRIEF_FULL.md §8 and Fix 3.

## Primary behavioral estimand (8B, later)

AME of INT4 vs FP16 on `follows_context`, double-filtered, **Llama-3.1-8B-Instruct**.

## Primary mechanistic estimand (8B, later)

Δ steering efficacy: (FP16-heads @ INT4) − (FP16-heads @ FP16) − random-head Δ.

## Secondary

Jaccard of reliable-head sets vs FP16 bootstrap 5th percentile.

## Repair estimand (Fix 1, 1B pilot on this repo)

Bootstrap CI on:

- re-ID-32 minus transferred
- rescale-only minus transferred

## Fix-3 triage order (before Stage 2 on 8B)

1. **Instruct = headline** (full repair grid).
2. **Base = confound check** (2×2 transfer only; repair grid only if time remains after instruct is closed).
3. If week budget for Stage 2 is tight: stop base at 2×2, finish instruct A/B/C.

## 1B pilot defaults (3050 Ti)

| Flag | Debug default | Scale-up target |
|------|---------------|-----------------|
| `--n_items` | 50 | 200–500 (mini-pilot), 32 (Fix 1 repair grid) |
| `--bootstrap_draws` | 5 | 20+ |
| `--K` | 5 | 5 (JuICE small-model default) |
| `--seed` | 42 | fixed across precisions |
