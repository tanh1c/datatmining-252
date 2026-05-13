# 0.68737 — SPECTER2 fine-tune, 5 seeds (regression from 0.69972)

Submitted: 2026-05-13
Public LB: **0.68737**
Anchor (3-seed run): `0.69972`
Difference: **−0.01235 public**, despite +0.0056 OOF.

This folder is **not** the current best. It is archived because:

1. The 5-seed run gave the highest **OOF QWK** so far (`0.6445`), so the
   averaged continuous scores remain useful as a **diversity feature** when
   stacking in step 5 (different threshold cut, but the underlying ranking is
   highly correlated with 0.69972).
2. The drop in public LB despite better OOF is an important data point about
   threshold tuning and label-distribution drift, recorded in
   `outputs/lessons_learned.md`.

## What changed vs 0.69972

| | 3-seed (0.69972) | 5-seed (this) |
| --- | ---: | ---: |
| Seeds | `[252, 253, 254]` | `[252, 253, 254, 255, 256]` |
| Models trained | 15 | 25 |
| OOF QWK (tuned) | 0.6389 | **0.6445** |
| OOF MAE | 0.8011 | **0.7831** |
| Public LB | **0.69972** | 0.68737 |
| Threshold 1 | **1.808** | 1.597 |
| Threshold 2 | 2.457 | 2.546 |
| Threshold 3 | 3.230 | 3.240 |
| Threshold 4 | 4.132 | 4.180 |
| Combined dist {1, 2, 3, 4, 5} | {252, 128, 102, 63, 51} | {180, **218**, 87, 61, 50} |
| Public split dist | {116, 64, 59, 33, 26} | {85, **106**, 49, 31, 27} |

## Why public got worse

Threshold 1 dropped from `1.808` to `1.597`. As a result, ~72 papers that the
3-seed model had assigned to label 1 were re-assigned to label 2 in the 5-seed
run. The training label distribution is roughly `{1: 36%, 2: 21%, 3: 18%, 4:
15%, 5: 11%}`. After threshold tuning:

- 3-seed public distribution: `{1: 39%, 2: 22%, 3: 20%, 4: 11%, 5: 9%}` —
  close to train distribution.
- 5-seed public distribution: `{1: 29%, 2: 36%, 3: 16%, 4: 10%, 5: 9%}` —
  label 2 is **inflated** to 36% (about 2x its expected proportion), label 1
  is suppressed.

`scipy.optimize.differential_evolution` is greedy on OOF QWK. It is allowed to
move thresholds anywhere inside the bounds with only a `min_gap >= 0.03`
constraint. With 5 seeds the OOF score distribution shifts slightly (more
averaging compresses variance), and the optimiser finds a slightly lower
threshold-1 that improves OOF by 0.006 but lands on a public-bad label
distribution.

## Saved artefacts

- `specter2_finetune_submission.csv` — what was submitted (Public LB 0.68737).
- `public_scores.csv`, `private_scores.csv` — averaged continuous scores per id
  across **25 models** (5 folds x 5 seeds). These are the per-id signals that
  stacking should use.
- `oof_scores.csv` — out-of-fold continuous scores for the 2494 train rows.
- `metrics.json` — full configuration + per-fold log.
- `step3b_specter2_finetune.ipynb` — the same notebook as 0.69972; only the
  `SEEDS` constant differs.

## How to use this folder downstream

When stacking in step 5:

- Use **0.69972 OOF + test scores** as the *primary* SPECTER2 anchor (public
  validated).
- Use **0.68737 OOF + test scores** as a *diversity* feature. The two score
  vectors are highly correlated (Pearson > 0.95 expected) but not identical;
  small differences come from the additional seeds 255 and 256.
- Tune threshold *constrained* (see `outputs/lessons_learned.md`) so a shifted
  OOF cannot push the predicted distribution far from the train distribution.
