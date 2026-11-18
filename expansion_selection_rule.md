# Stage 2 Evaluation-Pool Expansion Selection Rule

Status: preregistered selection only. No evaluation results are used by this rule.

## Source and deterministic procedure

- Source: `data/conflictqa_pilot250_popular.jsonl`
- Population: the full 250-item ConflictQA popular pool
- Shuffle: exactly `random.Random(42)`, matching `juice_repro/_make_stage2_nb.py`
- The source IDs are loaded in the source-file order, shuffled once, and indexed from zero.
- IDs are not filtered by closed-book correctness, model output, steering score, or any other outcome.

The resulting fixed positions are:

- Existing calibration set: positions `0:25` (25 items)
- Existing Stage 2 evaluation set: positions `25:75` (50 items; current n=50)
- n=100 extension: positions `75:125` (50 new items)
- n=150 extension: positions `125:175` (50 additional new items)

The n=100 set is the existing n=50 evaluation set plus the n=100
extension. The n=150 set is the n=100 set plus the n=150 extension.

## Model and eligibility distinction

This expansion draws from the full unfiltered 250-item pool using the same
seed and shuffle logic as the original Stage 2 split. It does **not** draw
from the Stage 1 Instruct 164-item closed-book pool. Stage 1 and Stage 2
use different model regimes (Instruct versus base) and different eligibility
criteria (closed-book double-filter versus no filter).

Stage 2 was never closed-book filtered. Consequently, the n=100 and n=150
extension items are not guaranteed to be closed-book correct at either
precision. This is consistent with the original n=50 selection logic and is
not a new limitation introduced by this expansion.

## Exact extension IDs

### n=100 extension (positions 75:125)

```text
ConflictQA-popQA-chatgpt_7623
ConflictQA-popQA-chatgpt_7907
ConflictQA-popQA-chatgpt_1627
ConflictQA-popQA-chatgpt_2022
ConflictQA-popQA-chatgpt_4922
ConflictQA-popQA-chatgpt_7706
ConflictQA-popQA-chatgpt_7826
ConflictQA-popQA-chatgpt_2631
ConflictQA-popQA-chatgpt_7587
ConflictQA-popQA-chatgpt_5609
ConflictQA-popQA-chatgpt_7637
ConflictQA-popQA-chatgpt_1325
ConflictQA-popQA-chatgpt_7696
ConflictQA-popQA-chatgpt_6644
ConflictQA-popQA-chatgpt_7794
ConflictQA-popQA-chatgpt_6105
ConflictQA-popQA-chatgpt_7663
ConflictQA-popQA-chatgpt_7787
ConflictQA-popQA-chatgpt_2665
ConflictQA-popQA-chatgpt_5337
ConflictQA-popQA-chatgpt_4183
ConflictQA-popQA-chatgpt_3213
ConflictQA-popQA-chatgpt_1447
ConflictQA-popQA-chatgpt_6111
ConflictQA-popQA-chatgpt_7647
ConflictQA-popQA-chatgpt_1353
ConflictQA-popQA-chatgpt_5947
ConflictQA-popQA-chatgpt_5720
ConflictQA-popQA-chatgpt_2329
ConflictQA-popQA-chatgpt_7861
ConflictQA-popQA-chatgpt_7153
ConflictQA-popQA-chatgpt_6387
ConflictQA-popQA-chatgpt_1252
ConflictQA-popQA-chatgpt_7568
ConflictQA-popQA-chatgpt_982
ConflictQA-popQA-chatgpt_2429
ConflictQA-popQA-chatgpt_7113
ConflictQA-popQA-chatgpt_3422
ConflictQA-popQA-chatgpt_6440
ConflictQA-popQA-chatgpt_5944
ConflictQA-popQA-chatgpt_4209
ConflictQA-popQA-chatgpt_2296
ConflictQA-popQA-chatgpt_7779
ConflictQA-popQA-chatgpt_2582
ConflictQA-popQA-chatgpt_4184
ConflictQA-popQA-chatgpt_2784
ConflictQA-popQA-chatgpt_1239
ConflictQA-popQA-chatgpt_1344
ConflictQA-popQA-chatgpt_7578
ConflictQA-popQA-chatgpt_7857
```

### n=150 extension (positions 125:175)

```text
ConflictQA-popQA-chatgpt_4667
ConflictQA-popQA-chatgpt_2485
ConflictQA-popQA-chatgpt_6162
ConflictQA-popQA-chatgpt_859
ConflictQA-popQA-chatgpt_4738
ConflictQA-popQA-chatgpt_1399
ConflictQA-popQA-chatgpt_404
ConflictQA-popQA-chatgpt_4335
ConflictQA-popQA-chatgpt_2207
ConflictQA-popQA-chatgpt_3104
ConflictQA-popQA-chatgpt_7614
ConflictQA-popQA-chatgpt_4345
ConflictQA-popQA-chatgpt_7675
ConflictQA-popQA-chatgpt_1863
ConflictQA-popQA-chatgpt_7797
ConflictQA-popQA-chatgpt_1398
ConflictQA-popQA-chatgpt_7793
ConflictQA-popQA-chatgpt_7692
ConflictQA-popQA-chatgpt_4829
ConflictQA-popQA-chatgpt_4196
ConflictQA-popQA-chatgpt_4486
ConflictQA-popQA-chatgpt_7714
ConflictQA-popQA-chatgpt_7724
ConflictQA-popQA-chatgpt_7656
ConflictQA-popQA-chatgpt_882
ConflictQA-popQA-chatgpt_6336
ConflictQA-popQA-chatgpt_7679
ConflictQA-popQA-chatgpt_7929
ConflictQA-popQA-chatgpt_7772
ConflictQA-popQA-chatgpt_529
ConflictQA-popQA-chatgpt_6142
ConflictQA-popQA-chatgpt_58
ConflictQA-popQA-chatgpt_5789
ConflictQA-popQA-chatgpt_7728
ConflictQA-popQA-chatgpt_3351
ConflictQA-popQA-chatgpt_890
ConflictQA-popQA-chatgpt_7609
ConflictQA-popQA-chatgpt_7831
ConflictQA-popQA-chatgpt_7665
ConflictQA-popQA-chatgpt_6507
ConflictQA-popQA-chatgpt_7749
ConflictQA-popQA-chatgpt_4382
ConflictQA-popQA-chatgpt_6338
ConflictQA-popQA-chatgpt_1940
ConflictQA-popQA-chatgpt_4750
ConflictQA-popQA-chatgpt_6331
ConflictQA-popQA-chatgpt_1924
ConflictQA-popQA-chatgpt_7596
ConflictQA-popQA-chatgpt_4442
ConflictQA-popQA-chatgpt_7775
```
