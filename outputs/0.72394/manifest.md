# 0.72394 — E5 4-anchor safest stack (current best)

Submitted: 2026-05-20
Public LB: **0.72394**
Previous public anchor: `0.72374` BGE safest stack
Jump: **+0.00020**

## Recipe

Per-id score:

```text
0.50 * e5 + 0.30 * scincl + 0.20 * specter2 + 0.00 * ridge
```

Then apply the distribution-constrained threshold tuner used by Step 13.

## Why this is staged as current best

This is the public-validated safest E5 stack. It improved public LB while keeping the combined test distribution close to train (`test_L1 = 0.1422`), which became the post-L17 safe-transfer reference band.

Later experiments did not replace it:

| Experiment | Local result | Promotion decision |
| --- | --- | --- |
| Fixed Qwen Step 15b | Qwen-bearing blends reached OOF about `0.6653` | rejected: no Qwen-bearing candidate stayed inside `test_L1 <= 0.147` |
| Scholarly graph Step 16 | graph/E5 correlation about `0.507` | rejected: standalone graph OOF only `0.3633`; safe graph blend lowered E5 OOF |

## Metrics

```text
OOF QWK (constrained tuner) = 0.6593
OOF round-QWK              = 0.6475
Test combined L1 vs train  = 0.1422
Public LB                  = 0.72394
```

## Files

- `next_e5_4anchor_safest_e5_submission.csv` — submitted file, Public LB 0.72394.
- `submission.csv` — same labels as the Step 13 safest E5 output folder.
- `oof_scores.csv` — train OOF score and thresholded prediction for this stack.
- `public_scores.csv` — public test continuous score and prediction.
- `private_scores.csv` — private test continuous score and prediction.
- `candidates.csv` — Step 13 E5 sweep candidates.
- `step13_e5_anchor_stack.py` — script used to generate the Step 13 E5 stack.

## Reproduce

```bash
python src/step13_e5_anchor_stack.py
```

Primary output:

```text
outputs/submissions/next_e5_4anchor_safest_e5_submission.csv
outputs/step13_e5_stack/safest_e5/
```
