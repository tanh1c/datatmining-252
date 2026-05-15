# 0.68718 — SPECTER2 fine-tune v2 (regression vs 0.69972)

Public LB: **0.68718** (anchor 0.69972, delta -0.01254). Despite OOF +0.006 and a constrained-tuner that brought OOF L1-distance to 0.0225.

Same pattern as 0.68737 (5-seed run). Higher OOF, lower public.

## Differences from 0.69972 (3b)
- Abstract cache: v2 (93.7%) -> v3 (95.7%)
- max_len: 256 -> 384
- batch_train: 16 -> 12
- Threshold tuner: unconstrained -> distribution-constrained (lambda=0.5)
- Seeds, folds: same

## OOF metrics
- OOF QWK constrained: 0.6446 (vs 3b 0.6389)
- OOF QWK unconstrained reference: 0.6478
- Thresholds (constrained): [1.885, 2.597, 3.339, 4.184]
- OOF L1 vs train dist: 0.0225 (vs 3b ~0.06)

## Test impact
- Combined dist: {1:261, 2:125, 3:100, 4:56, 5:54}
- 42 rows changed vs 0.69972: 29 down, 13 up
- Public split: 29 changes (21 down, 8 up) — most LB damage here
- Private split: 13 changes (8 down, 5 up)

## Why it failed
Three knobs changed at once. Even though constrained tuner kept OOF distribution near train, the *encoder representation itself changed* (v3 cache + max_len 384), so the test score distribution shifted and the same well-calibrated OOF thresholds did not transfer cleanly.

## Use as stacking diversity
- oof_scores.csv + public_scores.csv + private_scores.csv are still useful as diversity features in step 5 (different encoder config from 0.69972 and 0.68737).
