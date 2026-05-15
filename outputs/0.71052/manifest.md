# 0.71052 — Stacking blend (NEW BEST, beats SPECTER2 alone)

Submitted: 2026-05-15
Public LB: **0.71052**
Previous best: `0.69972` (SPECTER2 fine-tune 3-seed alone)
Jump: **+0.01080**

This is the first submission to break 0.70. The recipe is dead simple but it
required four prior failures (0.62820 lexical anchor, 0.63064 citation anchor,
0.68737 5-seed regression, 0.68718 v3+384 regression) to find.

## Method

Per-id score = `0.7 * specter_3s_oof_score + 0.3 * ridge_5x5_oof_score` for
the train OOF, and the same blend applied to public/private continuous scores.
Distribution-constrained threshold tuner (`lambda=0.5`) then maps the blended
score to labels 1-5.

Effectively: **70% semantic (SPECTER2 fine-tune) + 30% lexical (Ridge TF-IDF)**.

## Why exactly these two anchors

The stacking script (`outputs/stacking/`) tried 12 candidates against the same
4 anchors:

```
ridge_5x5    (lexical,   public 0.62820, OOF 0.5921)
specter_3s   (semantic,  public 0.69972, OOF 0.6373)  <-- public-best
specter_5s   (semantic,  public 0.68737, OOF 0.6403)  <-- correlated 0.993
specter_v2   (semantic,  public 0.68718, OOF 0.6446)  <-- correlated 0.981
```

Three SPECTER2 anchors are **highly correlated** (Pearson 0.98+) so they
add almost no diversity. Ridge correlates only 0.81 with SPECTER2 — that
*is* the genuine second signal.

Meta-models that use OOF to pick weights fall into a trap: they assign
specter_5s and specter_v2 high weights (slightly higher OOF) and **drop
specter_3s** (the public-best anchor). E.g. `huber_meta` gave specter_3s a
*negative* weight; `constrained_blend` gave it 0. Their OOF QWK looked good
(0.6534) but their **test L1 distribution distance** ballooned to 0.18,
predicting a public regression similar to 5-seed and v2 runs.

By dropping the redundant SPECTER2s and explicitly biasing toward the
public-validated anchor, `blend_2anchor_70_specter` ended up with:

- OOF QWK = 0.6464 (lower than huber_meta's 0.6534)
- **Test combined L1 distance = 0.152** (lowest among all candidates)
- Public LB = **0.71052**

The lower OOF QWK does not mean a worse model. Combined with the test L1
distance being the lowest among all candidates, this candidate produced a
predicted label distribution closest to the train distribution, which is the
proxy that actually correlates with public LB on this dataset.

## Files

- `blend_2anchor_70_specter_submission.csv` — what was submitted (Public LB 0.71052)
- `blend_2anchor_70_specter_oof.csv` — OOF blend score per train id
- `blend_2anchor_70_specter_public.csv` — public continuous scores + labels
- `blend_2anchor_70_specter_private.csv` — private continuous scores + labels
- `stacking_report.csv` — all 12 candidates ranked by OOF QWK and test L1
- `correlation_matrix.csv` — Pearson correlation across the 4 anchor OOFs
- `train_stacking_meta.py` — exact script used

## Per-anchor stats (with constrained tuner re-applied for fair comparison)

| Anchor | OOF QWK | Public LB |
| --- | ---: | ---: |
| ridge_5x5 | 0.5921 | 0.62820 |
| specter_3s | 0.6373 | **0.69972** |
| specter_5s | 0.6403 | 0.68737 |
| specter_v2 | 0.6446 | 0.68718 |

## Stack candidates ranked by test L1 (the better proxy for public)

| Candidate | OOF QWK | Test L1 | Public LB |
| --- | ---: | ---: | ---: |
| **blend_2anchor_70_specter** | **0.6464** | **0.152** | **0.71052** ✅ |
| simple_avg | 0.6508 | 0.162 | not submitted |
| blend_2anchor_80_specter | 0.6443 | 0.166 | not submitted |
| biased_specter3s_50_avg | 0.6486 | 0.172 | not submitted |
| blend_2anchor_60_specter | 0.6477 | 0.176 | not submitted |
| huber_meta | 0.6534 | 0.179 | not submitted |
| ridge_meta_a1 | 0.6514 | 0.179 | not submitted |

Note: this is the **third time** we see the pattern "higher OOF, more drift,
worse public" (after 0.68737 5-seed and 0.68718 v3+384). The constrained
tuner alone is not enough; we also need the *underlying score* to keep test
distribution close to train, and the simplest way is to bias toward the
anchor that is already public-validated.

## Reproduce

```bash
python train_stacking_meta.py
# outputs/stacking/blend_2anchor_70_specter_submission.csv == this
```

## Next directions

| Idea | Expected gain | Risk |
| --- | ---: | --- |
| Fine-tune **SciBERT** as a 4th truly-diverse anchor | +0.005 → +0.012 | low (different model = different correlation) |
| Add OpenAlex continuous OOF scores into the 2-anchor blend | +0.002 → +0.008 | low (need to re-run OpenAlex script to extract OOF) |
| Re-fit blend weight by tuning on test L1 + OOF QWK jointly | +0.003 → +0.005 | medium |
| LLM zero/few-shot ASP-relevance score as 5th anchor | +0.005 → +0.015 | medium |

The stacking ceiling is roughly bounded by the diversity of base anchors.
3 SPECTER2 fine-tunes + 1 Ridge is **2 effective signals**. Adding SciBERT
or LLM is the cleanest way to push higher.
