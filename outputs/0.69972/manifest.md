# 0.69972 — Best public score so far (SPECTER2 fine-tune)

Submitted: 2026-05-13
Public LB: **0.69972**
Previous best: 0.63064 (OpenAlex DOI Huber meta)
Jump: **+0.069**

This is the largest single jump in the project to date. It confirms the Step 1
hypothesis (Label = ASP/AI-symbolic topic relevance, not paper quality) and
shows that semantic understanding of `title + abstract` beats every TF-IDF /
metadata path tried before.

## Submission file

`specter2_finetune_submission.csv` (596 rows, ids ordered like
`data/raw/Test_Submission.csv`).

Test label distribution:

```
{1: 252, 2: 128, 3: 102, 4: 63, 5: 51}
```

Public split distribution: `{1: 116, 2: 64, 3: 59, 4: 33, 5: 26}`
Private split distribution: `{1: 136, 2: 64, 3: 43, 4: 30, 5: 25}`

Label-5 count = 51 (matches the 0.62820 anchor's 50 — no collapse).
Public/private label-5 are 26/25 — almost perfectly balanced, so the public
score should transfer cleanly to private.

## Method

**Step-by-step recipe to reproduce.** All artefacts are archived alongside this
manifest, plus the original notebook (`step3b_specter2_finetune.ipynb`) and the
exact abstract cache used as input (`abstracts_merged_v2.csv`).

### 1. Confirm the hypothesis

`src/eda_v2_keyword_signal.py` (Step 1 of the roadmap) showed that papers whose
titles contain ASP keywords (`asp`, `answer set`, `clingo`, `dlv`, `stable
model`, ...) have train-mean Label `3.92` vs `2.29` for the rest — a delta of
**+1.63**. Within each venue, the same pattern holds (e.g. +1.39 inside ICLP,
+1.54 inside CAV for `ai_symbolic` keywords). This proved the task is topic
relevance, not paper quality, so a semantic encoder of paper text is the right
direction.

### 2. Multi-source abstract crawl

`src/crawl_abstracts.py` (Step 2) pulled abstracts from four sources with
priority order `s2_abstract > s2_tldr > openalex_abstract > crossref_abstract >
openalex_title_abstract`. Final coverage:

- train: 2357 / 2494 = 94.5%
- public_test: 270 / 298 = 90.6%
- private_test: 268 / 298 = 89.9%

Per-source contribution (first non-empty wins):

- Semantic Scholar full abstract: 1916
- Semantic Scholar TLDR (AI-generated summary): 617
- OpenAlex abstract_inverted_index: 234
- OpenAlex title-search fallback: 137
- Crossref: 1

Cache file: `abstracts_merged_v2.csv` (also archived in this folder).

### 3. Fine-tune SPECTER2 with a regression head

The notebook `step3b_specter2_finetune.ipynb` did the actual training on
Google Colab A100. Configuration:

- Model: `allenai/specter2_base` (BERT-base, 110M params)
- Input: `title [SEP] abstract`, max 256 tokens
- Head: `Linear(768 -> 1)`, dropout 0.1
- Loss: SmoothL1Loss (Huber) on the float Label
- Optimizer: AdamW with separate learning rates
  - Encoder: `2e-5`, weight_decay `0.01`
  - Head: `1e-3`, weight_decay `0.01`
- Schedule: linear warmup `10%` then linear decay
- Mixed precision (fp16), grad clip `1.0`
- Epochs: 5, batch_train 16, batch_eval 64
- CV: StratifiedKFold(5) x seeds `[252, 253, 254]` -> 15 models
- Per-fold best epoch selected by validation round-QWK
- Test scores averaged across all 15 fold/seed models

Per-fold best round-QWK ranged 0.59-0.66; mean ~0.62.
A100 wall-clock: ~0.6 min/fold, ~10 minutes total.

### 4. Threshold tuning

Differential evolution on OOF scores tuned 4 thresholds:

```
[1.8080, 2.4573, 3.2297, 4.1316]
```

OOF metrics:

- QWK (tuned): **0.6389**
- MAE: 0.8011
- Macro-F1: 0.4034

The fourth threshold at 4.13 means the model is confidently mapping its
strongest scores to label 5 — a sharp contrast with the frozen-SPECTER2 attempt
in step 3a where the fourth threshold was 3.41 and only 14 rows were assigned
label 5.

## Why it works

1. SPECTER2 is pre-trained on scientific paper triples, so the encoder already
   has dense semantic features for ASP / Answer Set Programming, neural
   verification, neuro-symbolic AI, etc. that a TF-IDF model cannot capture
   (e.g. "ASP" ≡ "answer set programming" ≡ "stable model").
2. Fine-tuning on the labeled train set re-shapes that representation around
   the *label* axis instead of generic similarity. With 2494 labeled rows on a
   BERT-base sized encoder this is well within the regime where fine-tuning
   beats feature-extraction.
3. Repeated 5x3 CV with averaged test predictions reduces fold and seed
   variance, mirroring the 5x5 Ridge-ensemble pattern that was already
   public-safe.
4. The OOF threshold search prevents a label-distribution drift, which is what
   destroyed the frozen-SPECTER2 baseline.

## What did NOT work, and why we abandoned each direction

| Direction | Best public | Why dropped |
| --- | ---: | --- |
| RandomForest TF-IDF | 0.46096 | first valid baseline; sparse-tree ceiling |
| LogisticRegression word+char TF-IDF | 0.48-0.52 | helpful but plateau |
| LogReg `title + first-author surname`, tuned C | 0.52454 | metadata beats blind blends |
| XGBoost on sparse TF-IDF | 0.48093 | sparse + tree is wrong combination here |
| Ridge 5x5 + threshold (TF-IDF) | 0.62820 | proven anchor, kept for stacking |
| OpenAlex Huber meta over Ridge anchor | 0.63064 | citation features helped a little |
| Scholarly all-source dump | 0.60531 | external feature noise |
| COCI-only filtered scholarly | 0.61227 | OOF-stable, public-unstable |
| **Frozen SPECTER2 + Ridge (3a)** | not submitted | OOF 0.49, label-5 collapsed to 14 |
| **Fine-tuned SPECTER2 (3b, this folder)** | **0.69972** | semantic + fine-tune is the right combo |

## Next directions for >= 0.70 and beyond

1. **Stack** SPECTER2 fine-tune OOF with Ridge 0.62820 OOF and OpenAlex 0.63064
   OOF. Three near-uncorrelated public-validated signals (lexical, semantic,
   citation) usually add +0.01-0.03 on this kind of leaderboard.
2. Re-run the same notebook with **5 seeds** (`[252, 253, 254, 255, 256]`)
   instead of 3. Seed `254` gave the highest folds (`0.64-0.66`); two more
   seeds should push OOF to ~0.645-0.65 with no other change.
3. **LLM zero/few-shot ASP-relevance scoring** as an extra feature. Either via
   a small open model (Llama-3-8B / Mistral) or an API. Add it to the stack.
4. Try `allenai/specter2_aug2023refresh_base` (the more recent SPECTER2 base
   release). Same recipe, possibly +0.01 OOF.
5. Token length sweep: 256 -> 384 / 512. Some abstracts in this dataset are
   1500-3000 chars, so longer context may carry more signal at the cost of 2-3x
   training time.

## Files in this folder

- `specter2_finetune_submission.csv` — the actual submitted CSV.
- `public_scores.csv`, `private_scores.csv` — averaged continuous scores per id.
- `oof_scores.csv` — out-of-fold scores for the 2494 train rows.
- `metrics.json` — full configuration + metrics + fold log.
- `step3b_specter2_finetune.ipynb` — the exact notebook used.
- `abstracts_merged_v2.csv` — the abstract cache fed into the encoder.

This is the new primary anchor. Step 5 (stacking) should treat
`oof_scores.csv` + `public_scores.csv` + `private_scores.csv` as the strongest
signal and blend the older anchors as residual corrections.
