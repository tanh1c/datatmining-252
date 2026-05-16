# Current Best Method

Updated: 2026-05-16

## Best Public Score

- Public score: **`0.72103`**
- Submission: `outputs/0.72103/blend_3anchor_scincl_specter_ridge_60_20_20_submission.csv`
- Previous anchor: `outputs/0.71052/blend_2anchor_70_specter_submission.csv` scored `0.71052`
- Jump: **`+0.01051`** (first submission past 0.72)
- Archive: `outputs/0.72103/`

## Method Summary

A 3-anchor weighted stacking blend:

```
blended_score = 0.6 * scincl_oof + 0.2 * specter2_oof + 0.2 * ridge_tfidf_oof
```

Then a distribution-constrained threshold tuner (`lambda = 0.5 * |pred_dist -
train_dist|_1`) maps the blended score to labels 1-5.

Three anchors with three different signal axes:

- **SciNCL** (`malteos/scincl`): scientific BERT-base trained with
  neighborhood contrastive learning over the citation graph. Different
  pre-training objective than SPECTER2's triplet loss.
- **SPECTER2** (`allenai/specter2_base`): scientific BERT-base trained on
  paper-similarity triplets. Was the prior single-model best (0.69972).
- **Ridge TF-IDF**: lexical signal over `title_clean` + `authors_clean` +
  char n-grams. Captures author/venue/exact-token cues that BERT averages
  out.

## Pipeline (steps 1 → 6 in this repo)

```
data/raw/{train, public_test, private_test, Test_Submission}.csv
                |
                v
       Step 1: src/eda_v2_keyword_signal.py
        confirms Label = ASP / AI-symbolic relevance (proxy QWK 0.2589)
                |
                v
       Step 2: src/crawl_abstracts.py + crawl_abstracts_v2.py
        outputs/external/abstracts_merged_v2.csv (93.7% coverage,
        used by both step 3b and step 6)
                |
                v
       Step 3b: notebooks/step3b_specter2_finetune.ipynb (Colab A100)
        SPECTER2 base + Linear(768->1) head, SmoothL1, AdamW split LR,
        5 folds x 3 seeds, max_len 256, batch 16
        --> outputs/0.69972/{oof_scores, public_scores, private_scores}.csv
                |
                v
       Step 6: notebooks/step6_scincl_finetune.ipynb (Colab A100)
        SciNCL base + same head + same recipe, only the model name changes
        --> outputs/scincl_finetune/{oof_scores, public_scores, private_scores}.csv
                |
                v
       Older anchor: src/train_ridge_threshold_ensemble.py
        Ridge(alpha=8) over title + authors TF-IDF, 5 folds x 5 seeds
        --> outputs/ridge_threshold/reports/ridge_threshold_5fold_5seed_891e6ee4_*.csv
                |
                v
       Stacking: src/train_stacking_meta.py
        Auto-loads all available anchors, evaluates 40+ candidates,
        selected the 60/20/20 SciNCL/SPECTER2/Ridge weighted blend
        --> outputs/0.72103/blend_3anchor_scincl_specter_ridge_60_20_20_submission.csv
```

## Why this works

1. **Three orthogonal-ish signals.** SciNCL and SPECTER2 are both BERT-base
   trained on scientific corpora but with different contrastive objectives;
   Ridge captures lexical surface features that neither BERT can re-derive.
   Combined, they cover lexical, similarity-trained semantic, and
   neighbourhood-trained semantic axes.
2. **High weight on SciNCL (0.6) is unintuitive but works.** SciNCL has lower
   single-model OOF than SPECTER2 (0.6269 vs 0.6373) but its disagreement
   pattern with SPECTER2 falls on the right rows for the public split. With
   weight 0.6 on SciNCL, those disagreements drag the blend in the
   public-correct direction.
3. **Distribution-constrained tuner** prevents threshold drift (L1 dist
   penalty 0.5).
4. **Selecting candidate by lowest test-L1 distance within the SciNCL group**
   was the heuristic used. Even though both OOF and test L1 said "do not
   submit", the candidate within the new family with the closest predicted
   distribution to train turned out to transfer best (L9).

## OOF + test metrics

```
OOF QWK (constrained tuner) = 0.6412
OOF MAE                     = 0.7919
Predicted dist (combined)   = {1:255, 2:131, 3:101, 4:62, 5:47}
Test combined L1 vs train   = 0.162
```

Per-anchor reference (constrained tuner, same threshold recipe):

| Anchor | OOF QWK | Public LB alone |
| --- | ---: | ---: |
| ridge_5x5 | 0.5921 | 0.62820 |
| scincl | 0.6269 | not submitted alone |
| specter_3s | 0.6373 | 0.69972 |

Pairwise OOF Pearson correlation (the diversity matrix that drives stacking):

```
              ridge   specter  scincl  scibert  llm_zs
ridge          1.00     0.81    0.82    0.81    0.58
specter_3s     0.81     1.00    0.94    0.95    0.56
scincl         0.82     0.94    1.00    0.94    0.57
scibert        0.81     0.95    0.94    1.00    0.58
llm_zs         0.58     0.56    0.57    0.58    1.00
```

scincl/specter/scibert are all in the 0.94-0.95 cluster yet scincl + specter
gives more useful blending than scibert + specter. See L9 for why.

## Reproduce

```bash
# Pre-requisite: run steps 1, 2, 3b, 6 first so all anchor OOFs exist locally:
#   outputs/0.69972/oof_scores.csv (SPECTER2)
#   outputs/scincl_finetune/oof_scores.csv (SciNCL)
#   outputs/ridge_threshold/reports/ridge_threshold_5fold_5seed_891e6ee4_*.csv (Ridge)

python train_stacking_meta.py
# outputs/stacking/blend_3anchor_scincl_specter_ridge_60_20_20_submission.csv
# == outputs/0.72103/blend_3anchor_scincl_specter_ridge_60_20_20_submission.csv
```

## Anchors kept for future stacking

| Folder | Public LB | OOF QWK | Role |
| --- | ---: | ---: | --- |
| `outputs/0.62820/` | 0.62820 | 0.5921 | lexical (Ridge TF-IDF) |
| `outputs/0.69972/` | 0.69972 | 0.6373 | semantic SPECTER2 fine-tune |
| `outputs/0.71052/` | 0.71052 | 0.6464 | 2-anchor blend (SPECTER2 + Ridge) |
| `outputs/scincl_finetune/` | not submitted | 0.6269 | semantic SciNCL fine-tune |
| `outputs/scibert_finetune/` | not submitted | 0.6360 | semantic SciBERT (not useful in stack) |
| `outputs/0.72103/` | **0.72103** | 0.6412 | **3-anchor blend (current best)** ⭐ |

## Next directions ranked by expected value

| Idea | Effort | Expected gain | Risk |
| --- | --- | ---: | --- |
| **Weight sweep** around 60/20/20 (55/25/20, 65/15/20, 70/10/20) | low (1 min local + 1 daily slot) | +0.000 → +0.005 | low |
| **DeBERTa-v3-large fine-tune** as 4th anchor (different attention) | medium (~25 min Colab A100) | +0.005 → +0.015 | low |
| **bge-large-en-v1.5** fine-tune as 4th anchor (top MTEB embedder) | medium (~20 min Colab A100) | +0.003 → +0.012 | medium |
| Try **35/35/30** SciNCL blend (highest OOF among 3-anchors, untested) | low (already in stacking_report.csv, 1 daily slot) | +0.000 → +0.010 | medium |
| **SPECTER2 with `classification` adapter** as 5th anchor | low (1 dòng change in step 3b notebook) | +0.000 → +0.005 | low |

Top priority: weight sweep (cheap, tests local hypothesis).

## Failed earlier directions (kept for context)

| Submission / experiment | Public LB | Why dropped or failed |
| --- | ---: | --- |
| Frozen SPECTER2 + Ridge (3a) | not submitted | OOF 0.49, label-5 collapsed (L2) |
| 5-seed SPECTER2 (0.68737) | 0.68737 | distribution drift (L1) |
| v3 abstracts + max_len 384 (0.68718) | 0.68718 | three knobs at once (L6) |
| huber_meta stacking | not submitted | dropped public-best anchor by OOF (L7) |
| SciBERT fine-tune | not submitted | 0.948 corr with SPECTER2 + same OOF (no diversity) |
| LLM zero-shot (Qwen / DeepSeek) v1 + v2 | not submitted | OOF 0.37, signal too weak (L8) |
| All-source scholarly metadata blend | 0.60531 | external feature noise (L3) |
| Filtered COCI scholarly | 0.61227 | high OOF, public unstable (L3) |
