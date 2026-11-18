# Fix 1 pilot decision memo (A/B/C separability on 1B)

## Run metadata

- Date:
- bootstrap_draws:
- repair_n:

## Key numbers

- Transferred (FP16→INT4) efficacy:
- Rescale-only best α:
- Re-ID-32 efficacy:
- Negative control efficacy:
- Bootstrap CI overlap (unresolved if true):

## Hypothesis read

- A (identity drift): Jaccard below noise floor
- B (scale drift): rescale-only CI mostly above transferred
- C (rotation): rescale fails, re-ID succeeds, random fails

Decision (stop rule §6 Fix 1):
