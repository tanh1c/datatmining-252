# OpenAlex Title Search and Blend Follow-Up

## What Was Run

After pushing the `0.63064` checkpoint, the OpenAlex pipeline was extended in two ways:

1. Search OpenAlex by paper title for rows without DOI.
2. Create light blends between the `0.63064` OpenAlex DOI-only Huber model and the `0.62820` Ridge threshold anchor.

New reproducible command:

```bash
python train_openalex_title_meta.py
```

## Title-Search Coverage

OpenAlex source coverage after DOI + title search:

| Split | DOI match | Title match | No OpenAlex match |
| --- | ---: | ---: | ---: |
| train | 1455 | 742 | 297 |
| public_test | 177 | 100 | 21 |
| private_test | 182 | 93 | 23 |

Title-search cache:

```text
outputs/external/openalex_title_features.csv
```

## Title-Search Result

Best title-search meta candidate:

- Submission: `outputs/openalex_title_meta/submissions/openalex_title_huber_meta_w0.50_submission.csv`
- Convenience copy: `outputs/submissions/next_openalex_title_meta_submission.csv`
- Local OOF QWK: `0.603484`
- MAE: `0.856055`
- Test distribution: `{1:252, 2:124, 3:119, 4:45, 5:56}`

Interpretation:

- Title search greatly improves OpenAlex coverage.
- However, the best title-search model is below the DOI-only Huber candidate (`0.606685` OOF).
- Do not make this the primary submission before a small Kaggle probe; title matches may add noise despite increasing coverage.

## Private-Safe Blend Result

Best blend candidate:

- Submission: `outputs/private_safe_blends/submissions/blend_ridge062820_openalex063064_w0.65_submission.csv`
- Convenience copy: `outputs/submissions/next_private_safe_blend_submission.csv`
- Blend: `65%` OpenAlex DOI-only Huber + `35%` Ridge threshold anchor
- Local OOF QWK: `0.607028`
- MAE: `0.842021`
- Test distribution: `{1:198, 2:197, 3:114, 4:32, 5:55}`

Interpretation:

- This is the strongest local OOF score so far.
- It is a better next submission than the title-search candidate because it improves local validation while staying tied to both public-proven anchors.
- The main risk is that the distribution has fewer label `4` rows than the current `0.63064` submission.

## Recommended Submit Order

1. `outputs/submissions/next_private_safe_blend_submission.csv`
2. `outputs/submissions/next_openalex_title_meta_submission.csv`

If the private-safe blend improves public LB, continue blending around OpenAlex weight `0.60..0.75`. If it drops, keep `0.63064` as primary and treat title-search features as noisy until match quality is improved.
