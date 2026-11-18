# Whose Word Survives Compression?

**Working title:** *Do Causal Interventions for Context–Memory Arbitration Transfer Across Quantization?*

**Status:** Pre-pilot. Locked design as of 23 Aug 2026.
**Target:** ACL / EMNLP / ICLR **main**, Findings as the honest fallback, BlackboxNLP 2027 as the safety net.
**Compute:** lab RTX A4500 (20 GB) is the primary machine for all stages. RTX 3050 Ti = script debug only.
**This document** is the single source of truth: why this project exists, the exact research question, what we already ruled out and why, the full experimental design, the de-risking fixes on top of that design, and the week-by-week roadmap. If you read only one file, read this one.

---

## 1. The situation, in one page

You want a **top-tier NLP/ML paper** about a precise hole between two threads that have never been joined:

1. **How LLMs decide whose word to trust** when retrieved context contradicts parametric memory. The key paper is **JuICE** (Li, Chen & Tong, ICML 2025 Spotlight). It overturns the old story that some attention heads are "memory heads" and others are "context heads" (Jin et al. 2024). Instead, influential heads sit in **superposition** — the same head pushes both sources depending on the intervention. JuICE is a **test-time, training-free** intervention: identify a small set of "reliable" heads, run the model twice (once to record head outputs, once to inject scaled copies), and you can steer the model toward context or toward memory without fine-tuning.

2. **What quantization does to stored facts.** *Through a Compressed Lens* (Wang et al., TrustNLP @ ACL 2026) shows weight quantization (GPTQ / AWQ / bitsandbytes) quietly damages **factual recall**, more so in smaller models within a family. That paper has **no retrieved conflicting context** — it studies parametric memory in isolation.

**The locked research question:**

> If you compute JuICE's causal intervention on the FP16 checkpoint, does it still steer the *same* model after INT4 quantization? Or did quantization change the causal object (which heads, which direction, which gain) so the FP16 recipe silently misfires — even on items where the underlying fact itself still survives quantization?

This is **not** "does INT4 lower QA accuracy." Yazan et al. (2024) already tested quantization × RAG on non-conflicting retrieval and found a near-null on 7–8B-class models. Standard accuracy will not surface an arbitration failure — you need the conflict setup plus a causal probe.

**The sentence we are trying to earn (the main-track claim):**

> Quantization is not only a tax on facts. It is a transform of the residual-stream geometry that source-arbitration interventions were computed in. FP16-identified heads/axes are a different causal object after INT4, unless you re-identify or rescale them.

**If that sentence turns out false** (transfer works fine, Jaccard stays high, random heads don't mimic any drop), we still have a paper: *arbitration is more robust than recall* — a real qualification of *Compressed Lens*. That's Findings-shaped if the 8B evidence is tight, not automatically main-track.

---

## 2. The path — how this project got here (read this so we don't re-litigate)

Three design generations. Only Generation 3 is locked. Do not reopen Generations 1–2's questions.

### Generation 1 — naive (killed)
- Question: does quantization hit "context heads" harder than "memory heads"?
- **Fatal:** JuICE already refuted clean head types. A reviewer who has read ICML 2025 kills this in paragraph one.
- Tooling was GGUF/llama.cpp — fatal for Stage 2, no clean per-head hooks for the quantizers we care about.
- Venue (BlackboxNLP 2026) is calendar-dead.

### Generation 2 — workshop-honest (kept as ladder, not ambition)
- Question: do FP16 JuICE interventions transfer to INT8/INT4?
- Models: 0.6B–1.7B only (hardware-honest for a 3050 Ti).
- Good workshop paper. Not ACL/EMNLP main as written.

### Generation 3 — locked (this document)

| Upgrade from Gen 2 | Why |
|---|---|
| Double-filter (fact must survive at INT4 too) | Otherwise "more context-following" = the model just forgot the fact. That's *Compressed Lens*, not us. |
| 8B required (Llama-3.1-8B instruct **and** base) | Serving scale; JuICE wasn't a 0.6B paper; instruct vs. base is a known confound (Xie et al.). |
| 2×2 identify×apply, not transfer-only | An efficacy drop could be scale shift, noise, or capability collapse. Native INT4 JuICE vs. transferred JuICE is the real result. |
| Bootstrap Jaccard + random-head control | Without a noise floor, "heads drifted" is uninterpretable. |
| Repair = 32-shot re-ID / rescale, not TASA | TASA (mixed-precision allocation) is a second, systems-flavored paper. Re-ID is the interpretability "so what." |
| FP16 vs. INT4 primary; INT8 is a control, not a data point on a curve | 8-bit is often near-lossless. A pretty 3-point bit-width line is not a theory. |
| GPTQ only in Stage 1 | A grid is not a paper. |
| Honest claim scope: steering objects, not "all RAG pipelines break" | Production RAG doesn't run JuICE. Overclaiming dies in review. |

**Do not reopen:** head-type dichotomy, GGUF, BlackboxNLP 2026, 0.6B as a main-table star, multilingual as the flagship, safety×quant, building a new conflict benchmark.

---

## 3. Literature map (what each paper is *for*, not a reading list)

### Core — read fully, take notes

| Paper | Venue | Our use |
|---|---|---|
| Li, Chen, Tong. *Taming Knowledge Conflicts in Language Models.* | ICML 2025 Spotlight, PMLR 267 | **The method we transfer.** Superposition; reliable heads; dual-run JuICE. Code: github.com/GaotangLi/JUICE. Dataset: ParaConflict (`gaotang/ParaConfilct`). Their models: Gemma-2B, Llama2-7B, Llama3-8B, Phi-2.7B, StableLM2-1.6B, OLMo-7B — **base**, not instruct. |
| Jin et al. *Tug-of-War Between Knowledge.* | LREC-COLING 2024 | The **refuted** memory-head vs. context-head story. Cite as prior dichotomy. Do not build on it. |
| Xie et al. *Adaptive Chameleon or Stubborn Sloth.* | ICLR 2024 | Source of **ConflictQA**. Behavioral protocol for closed-book vs. conflicting context. Instruct models follow context more than expected — this is why base vs. instruct is locked in. |
| Su et al. *ConflictBank.* | 2024 | Larger conflict bench; **OAR** metric for comparability. |
| Wang et al. *Through a Compressed Lens.* | TrustNLP @ ACL 2026 | Quantization × **factual recall**, no conflict. The paper we either qualify or get absorbed by. Smaller models hurt more; 8-bit often OK; 4-bit messy. |
| Yazan et al. *Impact of Quantization on RAG.* | 2024 | Quant × RAG accuracy, mostly non-conflict. Our "accuracy won't catch this" citation. |

### Adjacent — read abstracts + relevant sections, do not pivot toward

| Paper | Why it's adjacent, not the project |
|---|---|
| Xu et al. EMNLP 2024, knowledge-conflict survey | Positioning paragraph only. |
| Wang et al. 2026, TASA (arXiv:2607.00908) | Perplexity-ranked sensitivity ≠ reasoning-critical layers. Discussion/appendix only, not Stage 3. |
| CLEAR (arXiv:2601.07041) | Multilingual ConflictQA. Optional robustness paragraph (incl. Bengali). **Not a Bangla paper.** |
| Language-Conditional Dequantization (arXiv:2608.11786) | Only relevant if the multilingual note happens. |
| Quant × safety/refusal (QuantiBias, Q-realign, Q-resafe, KV-cache alignment collapse) | Saturated. Explicitly out of scope. |
| Long-context quant (EMNLP 2025) | Different failure mode (length), not source arbitration. |

### Checked for novelty (as of Aug 2026)

No existing paper combines JuICE-style causal head steering with a quantization-transfer study. The two closest neighbors are:
- *Where Knowledge Collides* (arXiv 2601.09445) — a different steering method (CMAP) on knowledge conflict, no quantization.
- *Quantisation Reshapes the Metacognitive Geometry of Language Models* (arXiv 2604.08976) — quantization's effect on model confidence/calibration, no source-arbitration angle.

Cite both as close-but-distinct related work. Worth moving with real urgency — this is exactly the kind of gap someone else could land in independently.

### JuICE internals you must be able to explain out loud, no notes

- **Superposition:** a head can increase *both* P(parametric) and P(context) depending on intervention sign / conflict type. This is why knockout and linear probes lie about what a head "does."
- **Reliable heads:** heads whose *scaling* consistently moves the model toward one intended source on a small identification set.
- **Dual-run ("Just Run Twice"):** run 1 saves head outputs; run 2 adds scaled copies. Needed because a single-run scale still mixes the superposition.
- **K:** JuICE used K=5 reliable heads on smaller models, K=10 on 7–8B.
- **Conflict types:** clean / substitution / coherent. Coherent conflict is the hard case. Our 8B Stage 2 should use their types, not only ConflictQA entity swaps.

If you cannot explain those five bullets without notes, you are not ready to write Stage 2 code.

---

## 4. Concepts you must own (learning path, in order)

Do not "learn interpretability" in general. Learn this specific stack.

### Track 1 — Behavioral knowledge conflict
**Learn:** parametric vs. contextual knowledge; confirmation bias/chameleon vs. sloth framing; MR (memorization ratio) = fraction of conflict items where the answer matches the model's *own* closed-book answer; OAR from ConflictBank; why filtering on closed-book correctness is required.
**Done when:** you can look at one ConflictQA row, write the three prompts (closed-book, agrees, conflicts), and score a model output as parametric / context / other by hand.
**Pitfall:** "other" (refusal, mix, different entity) is not context-following. Keep a three-way label, not a binary one.

### Track 2 — Quantization as a change of function
**Learn:** weight-only PTQ; INT8 vs. INT4; GPTQ (calibration, Hessian-based error propagation); why the GPU still computes in FP16 on *dequantized* weights; bitsandbytes NF4 vs. GPTQ are not interchangeable; GGUF is a runtime format, not our analysis stack.
**Done when:** you can load Llama-3.2-1B-Instruct in transformers FP16 and GPTQ-INT4, greedy-decode the same 10 questions on both, and explain that hooks must see the *dequantized* matmul, not quantized weights directly.
**Pitfall:** comparing llama.cpp GGUF Q4_K_M to HF GPTQ and calling both "INT4." They are different functions.

### Track 3 — Causal interventions on attention
**Learn:** activation patching / hooks; how JuICE's dual-run differs from single-run steering; why bootstrap resampling of the identification set gives you a noise floor; Jaccard/rank overlap as a set-drift metric distinct from an efficacy-drift metric.
**Done when:** you can explain, in one paragraph, the difference between "the heads changed" and "the same heads now need a different scale."

---

## 5. What we will do — full experimental design

### 5.1 Mini-pilot (do this before any full Stage 1) — 3–4 days

**Goal:** decide if RQ0 (the forgetting confound) kills the project outright.

| Item | Spec |
|---|---|
| Model | Llama-3.2-1B-Instruct |
| Precision | FP16 vs. GPTQ-INT4 (same tokenizer, greedy, T=0, seed fixed) |
| Data | ConflictQA, filtered to items with a clear parametric target |
| Conditions | closed-book / context-agrees / context-conflicts |
| Metrics | closed-book exact match; MR on conflicts; MR **restricted to items correct closed-book at both precisions** (MR★) |

**Go if:** double-filtered MR still moves by ≳8–10 pp, **or** closed-book is stable but conflict behavior is not.
**Pivot if:** the MR move vanishes after the double filter (pure forgetting). Then either (a) kill the project, or (b) reframe as a *negative dissociation* paper and jump straight to 8B to see if the null holds at serving scale — a tight null on 8B is more publishable than a 1B confound.

**Do not** start JuICE reproduction on the exact same items until this plot exists — but see Fix 2 below for why the JuICE *environment* setup runs in parallel from day one.

### 5.2 Models × precisions (main table)

| Role | Model | Why |
|---|---|---|
| Screen | Llama-3.2-1B-Instruct | Fast kill / RQ0 gate |
| Screen | Qwen2.5/Qwen3-1.7B-Instruct | Second family, still small |
| **Required** | **Llama-3.1-8B-Instruct** | Serving scale; JuICE-adjacent family |
| **Required counterpart** | **Llama-3.1-8B (base)** | Separates the instruct confound from the quantization effect |
| Optional second 7–8B | Qwen2.5-7B-Instruct | Only if Llama 8B shows an effect |
| Appendix only | Gemma-3-1B-it, Qwen3-0.6B | Never in the main claim/abstract |

**Precisions:** FP16, INT8 (control), INT4 (test). **Quantizer:** GPTQ for all main cells; AWQ only on Llama-3.1-8B-Instruct if GPTQ transfer fails. **Decode:** greedy, T=0.
**Weight-only quantization note:** we intervene on the dequantized weights the model actually matmuls with. State this once in the paper — it's standard, reviewers will ask.

### 5.3 Data and filtering

Use existing benches; do **not** build a new conflict dataset — that's a different paper.

1. **ConflictQA** (Xie et al. / ICLR 2024) — parametric vs. counter-memory.
2. **ConflictBank** — report OAR for comparability.
3. **JuICE / ParaConflict splits** on the 8B model — Stage 2 is a *reproduction plus transform*, not a new task JuICE never saw.

**Item inclusion (main analysis set):**
```
keep item i for model M at precision p iff
  M_FP16 answers closed-book correctly
  AND M_p answers closed-book correctly
  AND context-conflicts passage is a true contradiction of that answer
```
Per-model sets will differ — that's fine, do not force a shared item set across model families. **Do** force a shared item set across precisions *within* one model, or you leak the confound.

**Confidence:** if sequence logprob of the parametric answer vs. context answer is available, use it. Binary EM is the headline; the logprob gap is the robustness plot.

### 5.4 Stage 1 — Behavioral, confound-first

**Metrics:** MR/OAR unfiltered (appendix); **MR★** on the double-filtered set (headline); parametric survival (the *Compressed Lens* replication on our items); agree-context sanity check (should stay high — if it collapses, we broke generation, not arbitration).

**Stats:** logistic mixed-effects `follows_context ~ precision + (1|item)` on the double-filtered set. Report **average marginal effect (AME) in percentage points** with a 95% CI, not only a p-value. Secondary: paired McNemar FP16 vs. INT4 per model, for comparability with ConflictBank/CLEAR. Pre-register: INT8 is expected near-null — don't treat "only INT4 moves" as a failure.

**Kill/pivot:**
- Effect only in 1B, gone at 8B → workshop/Findings-short, not main.
- Effect only unfiltered, gone after double filter → we've replicated *Compressed Lens*, nothing more. Stop, or write the dissociation paper if 8B closed-book is hurt but MR★ stays flat — that combination *is* interesting.
- Effect present in 8B on MR★ → proceed to Stage 2.

### 5.5 Stage 2 — Mechanistic core (this is the main-track section)

**Intervention:** JuICE dual-run, published protocol and code (github.com/GaotangLi/JUICE), nnsight per their environment.

**2×2 identify×apply table:**

| Heads identified on | Applied to | Quantity |
|---|---|---|
| FP16 | FP16 | Reproduction (must match JuICE closely on overlapping settings) |
| FP16 | INT4 | **Transfer** |
| INT4 | INT4 | **Native — does the circuit still exist?** |
| INT4 | FP16 | Reverse transfer |

**Plus controls:**

| Control | Why |
|---|---|
| Bootstrap identification at FP16 (resample calibration set, 20+ draws) | Noise floor for Jaccard. Cross-precision Jaccard must fall below this floor to mean anything. |
| Jaccard / rank overlap of reliable-head sets FP16 vs. INT4 | Separates set drift from efficacy drift. |
| Random-K heads, same scaling magnitude | If a random intervention also "breaks" at INT4, we've measured general brittleness, not a specific circuit failure. |
| Rescale-only repair | Grid a scalar α on 32 items. If this restores transfer, the axis exists and only the gain changed — a geometry/scale story, not "wrong heads." |
| Re-identify-32-shot repair | The actual cheap fix; replaces old Stage 3, the "so what" of the paper. |

**Mechanism to try to name, if transfer fails but native INT4 JuICE works:**
- **Hypothesis A — identity drift:** different heads are reliable. Jaccard low vs. FP16 bootstrap.
- **Hypothesis B — scale/gain drift:** same heads, wrong α. Rescale-only restores it.
- **Hypothesis C — superposition rotation:** same heads, direction in residual stream no longer aligned; rescale fails, re-identify succeeds, random fails.

Hypothesis C is the main-track sentence. A and B are still publishable if the 2×2 is clean — write them as such, do not overclaim C.

**Reproduction bar before any INT4 comparison:** JuICE on Llama-3.1-8B or Gemma-2B-class at FP16 must improve the intended source at a rate in the same ballpark as Li et al. — match *direction* and a clear gap vs. no-intervention, not every exact cell. If FP16 identification is already at chance on 1B, do not interpret 1B Jaccard — mechanistic claims live on 8B (and optionally Gemma-2B).

**Instruct vs. base:** run the 2×2 on instruct for the headline (serving); run FP16-vs-INT4 MR★ + transfer efficacy on base to show the effect isn't "chat template × quant."

### 5.6 Stage 3 — Repair (replaces mixed-precision TASA), only if transfer fails

Three equal-memory conditions on Llama-3.1-8B-Instruct INT4:
1. Uniform GPTQ-INT4, FP16 JuICE heads transferred (broken baseline).
2. Uniform GPTQ-INT4, **32-shot re-identified** JuICE heads (our repair).
3. Uniform GPTQ-INT4, **rescale-only** of FP16 heads (cheap ablation).

Success: (2) restores context-/memory-steering efficacy toward the FP16 JuICE number; (3) tells us whether new heads were actually needed.

TASA/perplexity-ranked mixed precision is a **related-work discussion point**, not the main experiment — unless repair (2) fails and compute remains. A reviewer prefers a clean 32-shot re-ID result over a half-baked mixed-precision system.

---

## 6. De-risking fixes on top of the design above

Three weak points decide whether this lands main-track or Findings. Fix each *before* it can quietly sink the project.

### Fix 1 — De-risk the repair experiment (Hypothesis A/B/C) — the real bottleneck

The A/B/C disambiguation happens on **n≈32 items**, on one 20GB card, at batch 1. If that signal is noisy, we cannot tell identity-drift from scale-drift from rotation — and without that separation we have "transfer fails" (Findings), not "transfer fails *because of X*" (main-track). This is not a wrap-up experiment; it's the load-bearing one.

**The fix:** run the full A/B/C protocol (identify → transfer → native → random-K → rescale-grid → re-ID-32) on **Llama-3.2-1B or Gemma-2B first**, before trusting it on 8B.
- If separation is clean on the cheap model → proceed to 8B with the same protocol, same code, same n.
- If A/B/C are not separable even on a cheap model → the problem is the protocol, not the scale. Fix it here (more re-ID shots, a second identification seed, a tighter rescale grid) before ever touching 8B.

**New diagnostics to add** (not in the original design, needed to trust this):
- Bootstrap CI on the repair delta itself (re-ID-32 minus transferred; rescale-only minus transferred). Heavily overlapping CIs mean A/B/C is unresolved — say so.
- Re-ID-32 on three different random 32-item draws. Large variance across draws means the "repair" is calibration noise, not a real re-identification signal.
- A negative control: re-identify on 32 random, task-irrelevant items. If this also "restores" steering, the repair isn't measuring what we think.

**Stop rule:** if the cheap-model pilot can't separate A/B/C with reasonable CIs, don't force a hypothesis name on 8B data alone. The paper becomes "transfer fails, mechanism unresolved" — still Findings-viable, but decide this before spending the 8B budget, not after.

### Fix 2 — Move JuICE FP16 reproduction earlier, run it in parallel

As originally sequenced, JuICE FP16 reproduction comes *after* the mini-pilot, the 1.7B replicate, and Stage 1's 8B behavioral work. If reproduction fails to match JuICE's direction, Stage 2 doesn't exist — and we'd find that out only after weeks of necessary-but-not-sufficient behavioral work.

**The fix — parallel track from Week 0:**

| Week | Mini-pilot track | JuICE reproduction track (parallel) |
|---|---|---|
| 0 | A4500 setup, 20-item smoke | Clone JuICE repo, pin `nnsight` env, run *their* unmodified script on *their* data — confirm the codebase works at all |
| 1 | 1B mini-pilot (MR vs. MR★) | JuICE FP16 on Gemma-2B or Llama-2-7B (their setup) — match direction only |
| 2 | Stage 1 8B behavioral starts | JuICE FP16 on **Llama-3.1-8B-base** (our actual target) — **gate** |

This splits two independent risks — "does the tooling work at all" vs. "does JuICE steer *our* target model" — so tooling failure surfaces in Week 0 and model-transfer failure surfaces by Week 2, before Stage 1 results pile up on a model we haven't confirmed is even steerable.

**Stop rule:** if JuICE does not steer Llama-3.1-8B-base in the expected direction by end of Week 2, either swap the mechanistic target to Gemma-2B (JuICE's own model — a weaker "serving-scale" claim) or drop to Stage-1-behavioral-only (workshop/Findings-short).

### Fix 3 — Pre-commit the base/instruct triage now, in writing

The required 8B arm is base **and** instruct (confound control), which roughly doubles Stage 2's surface area on hardware already running batch=1. Cutting one arm under a real week-9 deadline crunch means cutting it badly, exactly when clean results matter most.

**The fix — write this into `PRE_REG.md` before Stage 2 starts:**
1. **Instruct is the headline arm** (serving-relevant); **base is the confound check**. If time runs short, base gets cut *after* it has done its one job — confirming the Stage 1 effect isn't a chat-template artifact. Base does **not** need the full repair grid, only the 2×2 transfer table.
2. **Full repair protocol (Fix 1) runs on instruct only**, unless base shows something instruct doesn't — that would itself be worth the extra time.
3. **Trigger:** entering the week budgeted for "Stage 2 2×2 + bootstrap + random" without base's 2×2 table done → stop base at the 2×2 stage, skip its repair grid, and move all remaining time to closing out instruct's full A/B/C protocol.

Deciding this now means week 9 has no scope re-litigation under deadline pressure.

---

## 7. Roadmap (the actual week-by-week plan, fixes included)

Assumes **regular lab A4500 access**. If the box is shared, book multi-hour blocks for Phase 2–4; do scoring/write-up on the 3050 Ti between slots.

```
Week 0   A4500 setup + smoke        ‖  JuICE repo/env sanity (their data, their script)
Week 1   1B mini-pilot gate         ‖  JuICE FP16 on Gemma-2B/Llama-2-7B (their setup)
Week 2   Stage 1 8B behavioral start ‖ JuICE FP16 on Llama-3.1-8B-base (our target) — GATE
Week 3-4 Stage 1 done               ‖  Fix 1 pilot: full A/B/C protocol on 1B/Gemma-2B — GATE
Week 5-7 Stage 2 8B instruct: identify → transfer → native → random → rescale → re-ID
Week 8   Stage 2 8B base: 2×2 transfer table only (confound check)
Week 9   Triage checkpoint: base repair grid ONLY if instruct is fully done (Fix 3)
Week 10+ AWQ / second 7–8B only if GPTQ transfer failed and time remains
         CLEAR 2–3 languages only as a paragraph, only if everything above is done
Weeks 11-14 Write-up (ARR draft, three contributions max)
```

Two hard gates now sit *before* Stage 2 proper:
- **Week 2 gate:** JuICE actually steers our target model at FP16.
- **Week 4 gate:** the A/B/C repair protocol produces separable signal on a cheap model.

Failing either gate is cheap and informative this early — we'll know by Week 4 instead of Week 9, without having burned 8B-scale compute finding out.

**Rough end-to-end:** ~10–14 weeks experiment → draft, if lab access stays regular and the two gates pass on schedule. Compute budget: ~50–150 A4500-hours if disciplined; more if debugging nnsight×GPTQ. Always reproduce JuICE *unquantized* first — never skip that step to save time.

### Week 0 checklist (start here)
- [ ] `nvidia-smi` on lab PC; note driver/CUDA; confirm ~20 GB free under load
- [ ] HF licenses + Llama-3.2-1B-Instruct pulled to lab disk
- [ ] ConflictQA smoke JSONL (20 items) + scorer (parametric / context / other)
- [ ] FP16 vs. GPTQ-INT4 greedy loop on those 20 items, on the A4500
- [ ] JuICE repo cloned, pinned `nnsight` env imports working, on the lab machine
- [ ] Lab etiquette settled: disk quota, who to ask if jobs get killed, when the machine is free
- [ ] Clone JuICE, run *their* unmodified script on *their* data at FP16 (Fix 2, Week 0 track)

### Week 1 checklist
- [ ] Mini-pilot on hundreds of ConflictQA items (1B, A4500)
- [ ] Plot: parametric survival, MR, MR★
- [ ] Go/pivot/debug decision written in `writeup/mini_pilot.md`
- [ ] Pull Llama-3.1-8B (instruct now; base when Phase 3 starts) if go/pivot
- [ ] JuICE FP16 on Gemma-2B or Llama-2-7B — match their table's direction (Fix 2)

### Week 2 checklist
- [ ] JuICE FP16 on Llama-3.1-8B-base — this is the Fix-2 gate, do not skip
- [ ] `PRE_REG.md` filled with planned n, filter rule, and the Fix-3 triage order
- [ ] Start Stage 1 8B behavioral FP16 vs. INT4 (don't wait for the JuICE gate to fully close first — they can overlap)

### Weeks 3–4 checklist
- [ ] 8B instruct + base MR★ + AME plot in `writeup/`
- [ ] Fix 1 pilot: full A/B/C protocol on 1B or Gemma-2B, with bootstrap CIs on the repair delta — this is the second gate
- [ ] FP16 head ID + bootstrap Jaccard on ParaConflict (8B)
- [ ] Transfer apply FP16→INT4 (first Stage 2 cell, 8B)

Do not skip the decision memo after the mini-pilot, and do not skip the Fix 1 pilot decision memo either. The whole point of both gates is that **data can kill or pivot the claim**, and we obey that instead of inflating the grid or forcing a hypothesis name on noisy data.

---

## 8. Stats and pre-registration

Write a 1-page pre-reg (even if only in the repo) before the 8B run starts:
1. Primary behavioral estimand: AME of INT4 vs. FP16 on `follows_context`, double-filtered, Llama-3.1-8B-Instruct.
2. Primary mechanistic estimand: Δ steering efficacy, FP16-heads@INT4 minus FP16-heads@FP16, minus the random-head Δ.
3. Secondary: Jaccard vs. FP16 bootstrap 5th percentile.
4. Repair estimand: re-ID-32 vs. transferred heads on the same INT4 model, **with bootstrap CI (Fix 1)**.
5. Fix-3 triage order for base vs. instruct, written down before Stage 2 starts.

Software: `statsmodels` or R `lme4` for mixed models. Multiple comparisons: we have few *planned* contrasts — don't Bonferroni the appendix.

---

## 9. Publication ladder (unchanged honesty, higher ceiling)

| What we have | Venue |
|---|---|
| Mini-pilot / 1B only | Do not submit |
| Stage 1 8B behavioral, confound killed, no mechanism | Short / workshop / Findings-risky |
| Stage 1 + 2×2, transfer works (invariance result) | BlackboxNLP solid; Findings possible if 8B + tight null vs. *Compressed Lens* |
| Stage 1 + 2×2, transfer fails, native works, Jaccard below noise, random control holds | **Findings likely; main possible** |
| Above + Hypothesis C (rotation) named cleanly, or 32-shot repair that actually restores steering | **Main-track shaped** |
| Above + TASA bake-off | Extra, not required |

**ARR → ACL/EMNLP 2027** is the path. ICLR 2027/2028 if the geometry story (Hypothesis C) is the one that comes out. BlackboxNLP 2027 remains the safety net, not the ambition.

---

## 10. Writing the paper (for when results exist)

**Title pattern that matches the claim, not the method soup:**
- Good: *Quantization Breaks the Causal Object of Source Arbitration (Not Just the Facts)*
- Good: *FP16 Steering Directions Are Not Valid for INT4 Checkpoints*
- Bad: *A Comprehensive Empirical Study of Quantization, RAG, and Knowledge Conflicts in Small LLMs*

**Contribution list (max three):**
1. Confound-controlled evidence that source preference under conflict is/is not independent of fact survival under INT4.
2. Causal transfer evaluation of JuICE: heads and efficacy are/are not precision-invariant, with a stated noise floor.
3. A 32-shot re-identification (or rescale) that restores steering, showing the failure is *mis-specification of the intervention under the quantized function*, not "INT4 models cannot be steered."

**Related work to cite accurately:** Li et al. ICML 2025 (JuICE); Jin et al. 2024 (memory vs. context heads — the view JuICE refutes); Xie et al. ICLR 2024; ConflictBank; Wang et al. TrustNLP 2026 (*Compressed Lens*); Yazan et al. 2024 (quant × RAG, non-conflict); Wang et al. 2026 TASA (discussion only unless we run it); *Where Knowledge Collides* (arXiv 2601.09445) and *Quantisation Reshapes the Metacognitive Geometry* (arXiv 2604.08976) as closest-but-distinct neighbors.

---

## 11. What we will NOT do (so the paper stays one paper)

- No new conflict benchmark.
- No Bangla/multilingual flagship (CLEAR 2–3 languages as a single robustness paragraph, only if time remains).
- No safety/refusal × quant.
- No GGUF.
- No "context head vs. memory head" analysis.
- No 0.6B in the abstract or main table.
- No claiming production RAG pipelines use JuICE.
- No Stage-1 full GPTQ×AWQ×4-models grid.

---

## 12. Repository layout

```
quant-arbitration-study/
├── PROJECT_BRIEF_FULL.md     # this file — full context + design + fixes + roadmap
├── PRE_REG.md                 # 1-page estimands + Fix-3 triage order, written before 8B
├── data/
│   ├── conflictqa_filtered.jsonl
│   └── conflictbank_filtered.jsonl
├── mini_pilot/
│   └── run_1b_fp16_vs_int4.py
├── juice_repro/                # Fix 2: parallel track from Week 0
│   ├── their_setup_smoke.py
│   └── llama31_8b_base_fp16.py
├── stage1_behavioral/
│   ├── build_probe_set.py      # double closed-book filter
│   ├── run_inference.py
│   ├── score_mr.py             # MR, MR★, parametric survival
│   └── analyze_results.py      # AME + mixed model + McNemar
├── stage2_mechanistic/
│   ├── reproduce_juice_fp16.py
│   ├── identify_heads.py       # FP16 and INT4, + bootstrap
│   ├── apply_steering.py       # 2×2 transfer
│   ├── random_head_control.py
│   ├── head_drift_jaccard.py
│   └── rescale_and_reid.py     # repair — run on cheap model first (Fix 1)
└── writeup/
    ├── mini_pilot.md            # go/pivot decision memo
    ├── fix1_pilot.md            # A/B/C separability decision memo, cheap model
    └── draft.md
```

---

## 13. Daily/weekly operating rules

1. **One scientific question per week.** Week 1 is MR vs. MR★, not "also let's hook heads."
2. **Log every run:** model id, commit hash, quant method, seed, n items, filter rule. A result we can't regenerate isn't a result.
3. **Ban list:** GGUF, context-heads framing, 0.6B in the abstract, RAG-pipeline overclaim, new benchmark, Bangla pivot, TASA as the main fix.
4. **Stuck 48 hours on infrastructure?** Write the stall in `writeup/blockers.md` and change one variable only (loader, dtype, hook point).
5. **Don't expand the grid because a reviewer in your head asked.** RQ4 is optional scope. The paper is RQ0–RQ3.
6. **Negative results are allowed.** A tight 8B null with the full 2×2 beats a 1B p-hacked line plot.
7. **Instruct templates:** use the official chat template. Base models get the JuICE/Xie raw prompt. Mixing these is a silent confound.
8. **The Fix 1 and Fix 2 gates are not optional speed bumps.** Treat a failed gate as real information, not an obstacle to route around.

---

## 14. What "locked" means

**Locked:**
- Research question (transfer/native/geometry of JuICE under GPTQ INT4, confound-controlled).
- Venue ambition (main attempt, Findings likely, BlackboxNLP net).
- Model policy (1B screen, 8B required, base+instruct on Llama 8B).
- Stack (HF + GPTQ + nnsight).
- Primary GPU: lab RTX A4500 (20 GB). 3050 Ti = debug only.
- Stage 3 = re-ID/rescale, not TASA.
- Out-of-scope list (§11).
- The two Fix gates (Week 2 JuICE-steers-our-model; Week 4 A/B/C-separable-on-cheap-model).
- Fix-3 triage order: instruct is headline, base is confound-check-only under time pressure.

**Not locked (empirical, will move):**
- Whether MR★ moves.
- Whether Hypothesis A, B, or C is true.
- Whether we submit main vs. Findings vs. workshop.
- Exact n after filtering.

If a future version of this project wants to "just add AWQ everywhere" or "make it multilingual," that is a **new project**, not an edit to this one.

---

## 15. Glossary

| Term | Meaning here |
|---|---|
| Parametric / memory | Answer the weights would give with no passage |
| Context | Answer supported by the provided passage |
| Arbitration | Choosing between those two under conflict |
| MR | % of conflict items matching the model's own closed-book answer |
| MR★ | MR on the double-filtered set (headline) |
| Double filter | Correct closed-book at FP16 **and** at precision *p* |
| Reliable heads | JuICE identification output, not a fixed type-label |
| Transfer | Identify at FP16, apply at INT4 |
| Native | Identify and apply at the same precision |
| Causal object | The (heads, direction, scale) tuple that implements the intervention |
| Superposition | Same head contributes to both sources; JuICE's central empirical claim |
| AME | Average marginal effect, in percentage points, from the mixed-effects model |
| Fix 1 / Fix 2 / Fix 3 | The three de-risking patches in §6, applied on top of the locked design |
