# Current Best Method

Updated: 2026-05-13

## Best Public Score

- Public score: **`0.69972`**
- Submission: `outputs/specter2_finetune/specter2_finetune_submission.csv`
- Previous anchor: `outputs/openalex_meta/submissions/openalex_huber_meta_w1.00_submission.csv` scored `0.63064`
- Jump: **`+0.069`** (largest single-method jump in the project)
- Archive: `outputs/0.69972/`

## Method Summary

A SPECTER2 (`allenai/specter2_base`) encoder, **fine-tuned end-to-end** with a
linear regression head on the labeled train set, predicts a continuous
ASP-relevance score from `title [SEP] abstract`. The score is then mapped to
labels `1-5` via four thresholds tuned on OOF predictions.

The frozen-SPECTER2 + Ridge variant (Step 3a) collapsed to OOF `0.49` with only
14 label-5 predictions. Fine-tuning the encoder unlocks the topic-relevance
axis directly.

## Pipeline

The recipe is documented end-to-end in `outputs/0.69972/manifest.md`.

```
data/raw/{train,public_test,private_test,Test_Submission}.csv
                |
                v
       src/eda_v2_keyword_signal.py  (Step 1)
        confirms Label = ASP/AI-symbolic relevance
                |
                v
       src/crawl_abstracts.py        (Step 2)
        outputs/external/abstracts_merged_v2.csv  (93.7% coverage)
                |
                v
       notebooks/step3b_specter2_finetune.ipynb  (Step 3b, on Colab A100)
        - allenai/specter2_base + Linear(768->1)
        - SmoothL1Loss on float Label
        - AdamW (encoder 2e-5, head 1e-3), warmup 10%, fp16
        - 5 folds x 3 seeds, 5 epochs, max_len 256, batch 16
        - test predictions averaged across all 15 models
                |
                v
       differential_evolution threshold tune on OOF
                |
                v
       outputs/specter2_finetune/specter2_finetune_submission.csv
       Public LB: 0.69972
```

## Features used

Only two raw text fields per paper:

- `title`
- `abstract` (from `outputs/external/abstracts_merged_v2.csv`, priority
  `s2_abstract > s2_tldr > openalex_abstract > crossref_abstract >
  openalex_title_abstract`)

Not used in this candidate:

- venue, year, authors, doi (only used in older anchors)
- TF-IDF features
- citation / OpenAlex / Crossref metadata
- LLM scoring (Step 4 not done yet)

## Configuration

```python
MODEL_NAME    = 'allenai/specter2_base'
MAX_LEN       = 256
BATCH_TRAIN   = 16
BATCH_EVAL    = 64
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
SEP           = tokenizer.sep_token (i.e. [SEP])
INPUT_FORMAT  = f"{title}[SEP]{abstract}"  # title only if abstract empty
HEAD          = nn.Linear(768, 1) preceded by Dropout(0.1)
```

Per-fold best epoch selected by validation round-QWK. Test scores averaged
across all 15 fold-seed models.

## Thresholds

```
[1.808022956694, 2.457291915377, 3.229686586629, 4.131560616874]
```

Tuned by `scipy.optimize.differential_evolution` on OOF scores with a
minimum-gap penalty (`>= 0.03`) to avoid degenerate clusters.

## OOF metrics

```
QWK         = 0.6389
MAE         = 0.8011
Macro-F1    = 0.4034
```

Per-fold best round-QWK range `0.59 - 0.66`, mean `0.62`. Std across folds is
small (~0.02), suggesting the model trains stably.

## Test label distribution

Combined `{1: 252, 2: 128, 3: 102, 4: 63, 5: 51}`.

Public split `{1: 116, 2: 64, 3: 59, 4: 33, 5: 26}`.
Private split `{1: 136, 2: 64, 3: 43, 4: 30, 5: 25}`.

The 26/25 split of label-5 predictions across public/private is almost
perfectly balanced, suggesting the public LB result should transfer cleanly to
private.

## Why It Works

1. **Pre-trained scientific knowledge.** SPECTER2 was pre-trained on
   scientific paper triples, so its encoder already maps domain terminology
   like "ASP", "answer set programming", "stable model", "neuro-symbolic" to
   nearby vectors. TF-IDF cannot do this.
2. **Fine-tuning re-shapes the representation.** With 2494 labeled rows, the
   SmoothL1 regression objective shifts the encoder's [CLS] axis toward the
   ASP-relevance dimension. Frozen SPECTER2 + Ridge could not do this; it
   reached only OOF `0.49`.
3. **Repeated CV averaging.** 5 folds x 3 seeds = 15 models with averaged test
   scores reduces fold and seed variance, mirroring the public-safe 5x5 Ridge
   pattern.
4. **OOF threshold search.** The fourth threshold `4.13` lets the model push
   high-confidence ASP papers cleanly into label 5. Frozen SPECTER2 only
   reached threshold `3.41` and predicted 14 label-5 rows; the fine-tuned
   version predicts 51, matching the train distribution shape.

## Reproduce

1. Crawl abstracts (Step 2) if `outputs/external/abstracts_merged_v2.csv` is
   missing:

   ```bash
   python crawl_abstracts.py
   ```

2. Build the upload zip:

   ```bash
   python make_colab_zip.py
   ```

3. Open `notebooks/step3b_specter2_finetune.ipynb` on Google Colab with an
   A100 (or T4 with `BATCH_TRAIN=8`).
4. Cell 4 prompts for `asp_data.zip` upload.
5. Run all cells. Cell 10 downloads `specter2_finetune_outputs.zip`.
6. Extract under `outputs/specter2_finetune/` in the local repo.
7. Submit `outputs/specter2_finetune/specter2_finetune_submission.csv`.

Wall clock: ~10 minutes on A100, ~30-40 minutes on T4.

## Next directions

The roadmap to push further:

| Idea | Expected gain | Risk |
| --- | ---: | --- |
| **Stacking (Step 5)**: blend SPECTER2 OOF + Ridge `0.62820` OOF + OpenAlex `0.63064` OOF | **+0.01 to +0.03** | low — three near-uncorrelated signals |
| **5-seed rerun** of the same notebook | +0.005 to +0.010 | low — pure variance reduction |
| **LLM scoring (Step 4)** as a meta feature | +0.005 to +0.015 | medium — depends on LLM coverage |
| **max_len 384 or 512** (longer abstracts) | +0.005 to +0.010 | low — more compute |
| **`specter2_aug2023refresh_base`** model | +0.005 to +0.010 | low |
| Fine-tune `scibert_scivocab_uncased` for diversity | +0.005 (in stack) | low |

## Failed Earlier Directions

| Submission | Public LB | Local note |
| --- | ---: | --- |
| `outputs/submissions/next_scholarly_meta_submission.csv` | `0.60531` | all-source citation feature dump was noisy |
| `outputs/submissions/next_filtered_scholarly_submission.csv` | `0.61227` | COCI-only filtered features overfit local OOF |
| Frozen SPECTER2 + Ridge (3a) | not submitted | OOF 0.49, label-5 collapsed to 14 rows |

The lesson from these failures: external features and frozen embeddings are
brittle without the right model on top. Fine-tuning a well-chosen scientific
encoder is the cleaner path.

## Anchors kept for stacking

These older anchors are kept to provide diversity in any future ensemble:

- `outputs/0.62820/` — Ridge 5x5 + threshold (lexical anchor)
- `outputs/0.63064/` — OpenAlex Huber meta (citation anchor)
- `outputs/0.69972/` — **SPECTER2 fine-tune (semantic anchor, current best)**
