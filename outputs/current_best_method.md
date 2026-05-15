# Current Best Method

Updated: 2026-05-15

## Best Public Score

- Public score: **`0.71052`**
- Submission: `outputs/0.71052/blend_2anchor_70_specter_submission.csv`
- Previous anchor: `outputs/0.69972/specter2_finetune_submission.csv` scored `0.69972`
- Jump: **`+0.01080`** (first submission past 0.70)
- Archive: `outputs/0.71052/`

## Method Summary

A 2-anchor stacking blend:

```
blended_score = 0.7 * specter2_finetune_3seed_score + 0.3 * ridge_tfidf_5x5_score
```

Then a distribution-constrained threshold tuner (`lambda=0.5 * L1` on the
predicted vs train label distribution) maps the blended score to labels 1-5.

The two anchors are **truly different signals**:

- `specter_3s`: SPECTER2 fine-tuned end-to-end on `title + abstract`.
  Captures **semantic** ASP/AI-symbolic relevance.
- `ridge_5x5`: Ridge TF-IDF on `title_clean + authors_clean`, repeated 5x5
  CV ensemble. Captures **lexical** signals plus first-author surname leakage.

Pearson correlation between the two anchor OOFs: **0.811** — different enough
to add real value when combined.

## Pipeline (steps 1 -> 5 in this repo)

```
data/raw/{train,public_test,private_test,Test_Submission}.csv
                |
                v
       Step 1: src/eda_v2_keyword_signal.py
        confirms Label = ASP/AI-symbolic relevance (proxy QWK 0.2589)
                |
                v
       Step 2: src/crawl_abstracts.py + src/crawl_abstracts_v2.py
        outputs/external/abstracts_merged_v2.csv  (93.7% coverage)
                |
                v
       Step 3b: notebooks/step3b_specter2_finetune.ipynb (Colab A100)
        SPECTER2 base + Linear(768->1) head, SmoothL1, AdamW split LR,
        5 folds x 3 seeds, max_len 256, batch 16
        --> outputs/0.69972/{oof_scores, public_scores, private_scores}.csv
                |
                v
       Older anchor: src/train_ridge_threshold_ensemble.py
        Ridge(alpha=8) over title+authors TF-IDF, 5 folds x 5 seeds
        --> outputs/ridge_threshold/reports/ridge_threshold_5fold_5seed_891e6ee4_*.csv
                |
                v
       Step 5: src/train_stacking_meta.py
        loads both anchor OOFs and test scores, tries 12 blend candidates,
        picks the one with lowest test L1 distance and OOF QWK >= base
        --> outputs/0.71052/blend_2anchor_70_specter_submission.csv
```

## Why 70/30 specifically

The script tested simple_avg (0.5/0.5/0.5/0.5 over 4 anchors), weighted
averages, ridge_meta, huber_meta, constrained_blend, and 2-anchor blends at
50/50, 60/40, 70/30, 80/20.

- huber_meta and constrained_blend gave specter_3s (the public-best anchor)
  near-zero or negative weight because the *other* SPECTER2 variants had
  marginally higher OOF QWK. Their test L1 distance ballooned to 0.18.
- simple_avg over the 4 anchors was the 2nd-safest (test L1 0.162).
- **70% specter_3s + 30% ridge** had the lowest test L1 (0.152) of all
  candidates, confirming the predicted label distribution stayed close to
  the train distribution.

## Configuration of each anchor

### specter_3s (semantic anchor)
```python
MODEL_NAME    = 'allenai/specter2_base'
MAX_LEN       = 256
BATCH_TRAIN   = 16
EPOCHS        = 5
LR_ENCODER    = 2e-5
LR_HEAD       = 1e-3
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
GRAD_CLIP     = 1.0
LOSS          = SmoothL1Loss(beta=1.0)
PRECISION     = fp16
FOLDS         = 5
SEEDS         = [252, 253, 254]
INPUT_FORMAT  = f"{title}{tokenizer.sep_token}{abstract}"
HEAD          = nn.Linear(768, 1) preceded by Dropout(0.1)
```

### ridge_5x5 (lexical anchor)
```python
TF-IDF: title word ngram (1,2) + title char_wb ngram (3,5) + authors word ngram (1,2)
Ridge(alpha=8.0, solver='lsqr')
StratifiedKFold(5) x seeds [252..256] = 25 models
test scores averaged across all 25 models
```

### Threshold tuner (constrained)
```python
bounds = [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)]
lambda_dist_penalty = 0.5
objective = -QWK + 5 * gap_penalty + 0.5 * |pred_dist - train_dist|_1
optimiser = scipy.optimize.differential_evolution(maxiter=120, popsize=15)
```

Tuned thresholds: `[1.916, 2.457, 3.105, 3.864]`.

## OOF metrics

```
OOF QWK (constrained tuner) = 0.6464
OOF MAE                     = 0.7843
OOF macro-F1                = ~0.41
```

Per-anchor reference (constrained tuner):
- ridge_5x5: OOF 0.5921
- specter_3s: OOF 0.6373

So the blend gains **+0.009 OOF** over the strongest single anchor with the
same threshold recipe.

## Test label distribution

- Combined: `{1: 252, 2: 132, 3: 103, 4: 59, 5: 50}`
- Public: 153 (51%) of 298 are split between labels 1 and 2.
- L1 distance to train distribution (combined): **0.152** — best among all
  12 stack candidates.

## Why It Works

1. **Two diverse signals.** SPECTER2 captures domain semantics that TF-IDF
   misses ("answer set programming" ≡ "ASP" ≡ "stable model"). TF-IDF
   captures author/venue/exact-token signals that SPECTER2 averages out.
   Pearson correlation 0.81 is exactly the sweet spot for stacking — high
   enough that both anchors are talking about the same task, low enough that
   blending reduces variance.
2. **70/30 not 50/50.** The semantic anchor is stronger (public 0.69972 vs
   0.62820), so the blend should reflect that. 70/30 is calibrated by the
   public-LB ratio and validated by test L1 distance.
3. **Distribution-constrained tuner.** Threshold search would otherwise
   collapse into degenerate distributions per L1+L6 lessons.
4. **Selecting candidate by test L1 distance**, not OOF QWK, is the key step
   that prevented the OOF-overfit trap (L7 in `outputs/lessons_learned.md`).

## Reproduce

```bash
# 1. Make sure step 3b output is in outputs/0.69972/
# 2. Make sure ridge OOF is in outputs/ridge_threshold/reports/...891e6ee4...

python train_stacking_meta.py
# Best candidate auto-selected: outputs/stacking/best_submission.csv
# == outputs/0.71052/blend_2anchor_70_specter_submission.csv
```

## Anchors kept for future stacking

| Folder | Public LB | OOF QWK | Role |
| --- | ---: | ---: | --- |
| `outputs/0.62820/` | 0.62820 | 0.5921 | lexical (Ridge TF-IDF) ✅ |
| `outputs/0.63064/` | 0.63064 | 0.6067 | citation (OpenAlex) — OOF not yet extracted |
| `outputs/0.69972/` | 0.69972 | 0.6373 | semantic 3-seed ✅ (primary) |
| `outputs/0.68737/` | 0.68737 | 0.6403 | semantic 5-seed (correlated 0.99 with 3-seed, low diversity) |
| `outputs/0.68718/` | 0.68718 | 0.6446 | semantic v3+384 (correlated 0.98 with 3-seed) |
| `outputs/0.71052/` | **0.71052** | 0.6464 | **stacking blend (current best)** ⭐ |

## Next directions (ranked by expected gain)

| Idea | Effort | Expected gain | Risk |
| --- | --- | ---: | --- |
| Fine-tune **SciBERT** as a 4th truly diverse anchor | medium (1 Colab run) | +0.005 -> +0.012 | low |
| Extract OpenAlex Huber meta OOF and add to stack | medium (re-run script) | +0.002 -> +0.008 | low |
| LLM zero/few-shot ASP-relevance score | medium-high | +0.005 -> +0.015 | medium |
| Blend weight sweep 0.65/0.35, 0.75/0.25, 0.80/0.20 | low | +0.000 -> +0.005 | low |

The stacking ceiling is bounded by the **diversity** of base anchors.
We have 2 effective signals (semantic + lexical) — adding a 3rd genuinely
different signal (SciBERT or LLM scoring) is the cleanest path forward.

## Failed earlier directions (kept for context)

| Submission | Public LB | Why dropped |
| --- | ---: | --- |
| `next_scholarly_meta_submission.csv` | 0.60531 | all-source citation feature dump (L3) |
| `next_filtered_scholarly_submission.csv` | 0.61227 | COCI-only had high OOF, public unstable (L3) |
| Frozen SPECTER2 + Ridge (3a) | not submitted | OOF 0.49, label-5 collapsed to 14 (L2) |
| 5-seed SPECTER2 (0.68737) | 0.68737 | distribution drift (L1) |
| v3 abstracts + max_len 384 (0.68718) | 0.68718 | three knobs at once (L6) |
| huber_meta stacking | not submitted | dropped public-best anchor by OOF (L7) |
