import json
import re
import numpy as np

data = json.load(open('last-dance/gemma_clean_k2_int4_transfer_n100_ncalib128.json', encoding='utf-8'))

def looks_coherent(text: str) -> bool:
    t = (text or '').strip()
    if len(t) < 2: return False
    if t.count('://') >= 3: return False
    if re.fullmatch(r'[\W\d_]+', t): return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1: return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))

print("=== CONDITIONS SUMMARY ===")
for cond in ['fp16_clean_k2', 'int4_clean_k2', 'fp16_random_k2', 'int4_random_k2']:
    c = data[cond]
    rows = c['rows']
    b_labels = [r['baseline_qa_label'] for r in rows]
    s_labels = [r['steered_qa_label'] for r in rows]
    
    b_counts = {k: b_labels.count(k) for k in ['parametric', 'context', 'rejects_context', 'other']}
    s_counts = {k: s_labels.count(k) for k in ['parametric', 'context', 'rejects_context', 'other']}
    
    b_coh = sum(looks_coherent(r['baseline_text']) for r in rows) / len(rows)
    s_coh = sum(looks_coherent(r['steered_text']) for r in rows) / len(rows)
    
    print(f"\nCondition: {cond}")
    print(f"  Baseline fc: {c['fc_baseline']:.2f}")
    print(f"  Steered fc (inclusive): {c['fc_steered_inclusive']:.2f}")
    print(f"  Steered fc (gold-PAR-only): {c['fc_steered_gold_par_only']:.2f}")
    print(f"  Inclusive Delta:    {c['inclusive']['fmt']} (excl 0: {c['inclusive']['excludes_zero']}) [raw: {c['inclusive']['ci95_pp']}]")
    print(f"  Gold-PAR Delta:     {c['gold_par_only']['fmt']} (excl 0: {c['gold_par_only']['excludes_zero']}) [raw: {c['gold_par_only']['ci95_pp']}]")
    print(f"  n_gold_par_from_ctx: {c['n_gold_par_from_context']}, n_leave_without_par: {c['n_leave_ctx_without_par']}")
    print(f"  Baseline P/C/R/O: {b_counts['parametric']}/{b_counts['context']}/{b_counts['rejects_context']}/{b_counts['other']}")
    print(f"  Steered  P/C/R/O: {s_counts['parametric']}/{s_counts['context']}/{s_counts['rejects_context']}/{s_counts['other']}")
    print(f"  Coherence: baseline={b_coh*100:.1f}%, steered={s_coh*100:.1f}%")

# Compute Paired Specificity Contrast (id_delta - rand_delta)
print("\n" + "="*50)
print("PAIRED SPECIFICITY CONTRASTS & DiD")
print("="*50)

# Extract per-item binary delta for FP16 and INT4
# In our convention: delta = steered - baseline (for context following, so negative means leave context/enlarge parametric)
# Or in terms of parametric enlarge (following parametric / leaving context):
# Let's check both conventions: delta_fc (change in follows_context) and delta_enlarge (- delta_fc)

for mode in ['inclusive', 'gold_par_only']:
    print(f"\n--- Specificity Mode: {mode} ---")
    # FP16
    fp16_id_rows = data['fp16_clean_k2']['rows']
    fp16_rand_rows = data['fp16_random_k2']['rows']
    # INT4
    int4_id_rows = data['int4_clean_k2']['rows']
    int4_rand_rows = data['int4_random_k2']['rows']
    
    n = len(fp16_id_rows)
    rng = np.random.default_rng(42)
    
    # Per-item deltas for follows_context
    # id delta = steered_fc - baseline_fc
    key = 'steered_fc_inclusive' if mode == 'inclusive' else 'steered_fc_gold_par_only'
    
    fp16_d_id = np.array([r[key] - r['baseline_fc'] for r in fp16_id_rows], dtype=float)
    fp16_d_rand = np.array([r[key] - r['baseline_fc'] for r in fp16_rand_rows], dtype=float)
    
    int4_d_id = np.array([r[key] - r['baseline_fc'] for r in int4_id_rows], dtype=float)
    int4_d_rand = np.array([r[key] - r['baseline_fc'] for r in int4_rand_rows], dtype=float)
    
    # Specificity = delta_id - delta_rand (paired across items)
    fp16_spec = fp16_d_id - fp16_d_rand
    int4_spec = int4_d_id - int4_d_rand
    did = int4_spec - fp16_spec
    
    # Bootstrap
    fp16_spec_draws = []
    int4_spec_draws = []
    did_draws = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        fp16_spec_draws.append(100.0 * np.mean(fp16_spec[idx]))
        int4_spec_draws.append(100.0 * np.mean(int4_spec[idx]))
        did_draws.append(100.0 * np.mean(did[idx]))
        
    fp16_pt = 100.0 * np.mean(fp16_spec)
    fp16_ci = [np.percentile(fp16_spec_draws, 2.5), np.percentile(fp16_spec_draws, 97.5)]
    
    int4_pt = 100.0 * np.mean(int4_spec)
    int4_ci = [np.percentile(int4_spec_draws, 2.5), np.percentile(int4_spec_draws, 97.5)]
    
    did_pt = 100.0 * np.mean(did)
    did_ci = [np.percentile(did_draws, 2.5), np.percentile(did_draws, 97.5)]
    
    print(f"FP16 Specificity (id - rand): {fp16_pt:+.1f} [{fp16_ci[0]:+.1f}, {fp16_ci[1]:+.1f}] (excl 0: {not (fp16_ci[0] <= 0 <= fp16_ci[1])})")
    print(f"INT4 Specificity (id - rand): {int4_pt:+.1f} [{int4_ci[0]:+.1f}, {int4_ci[1]:+.1f}] (excl 0: {not (int4_ci[0] <= 0 <= int4_ci[1])})")
    print(f"DiD (INT4_spec - FP16_spec):  {did_pt:+.1f} [{did_ci[0]:+.1f}, {did_ci[1]:+.1f}] (excl 0: {not (did_ci[0] <= 0 <= did_ci[1])})")

print("\n" + "="*50)
print("FLIP INSPECTION ACROSS CONDITIONS")
print("="*50)
for cond in ['fp16_clean_k2', 'int4_clean_k2', 'fp16_random_k2', 'int4_random_k2']:
    rows = data[cond]['rows']
    flips = [r for r in rows if r['baseline_fc'] != r['steered_fc_inclusive']]
    print(f"\n=== {cond}: {len(flips)} flips ===")
    for f in flips:
        direction = "leave_context" if f['baseline_fc'] and not f['steered_fc_inclusive'] else "enter_context"
        print(f"  [{f['id']}] {direction}: {f['baseline_qa_label']} -> {f['steered_qa_label']} (mentions_gold_par={f['steered_mentions_par_gold']})")
        print(f"    Q: {f['question']}")
        print(f"    Gold PAR: {f['parametric_gold']} | Gold CTX: {f['context_gold']}")
        print(f"    Base: {f['baseline_text']!r}")
        print(f"    Ste : {f['steered_text']!r}")

