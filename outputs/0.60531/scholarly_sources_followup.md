# Scholarly Sources Follow-Up

## What Was Added

New reproducible command:

```bash
python train_scholarly_meta.py
```

The script combines existing OpenAlex caches with three new external sources:

- Semantic Scholar Graph API
- Crossref REST API
- OpenCitations COCI API

Source docs:

- Semantic Scholar: `https://api.semanticscholar.org/api-docs/`
- Crossref: `https://www.crossref.org/documentation/retrieve-metadata/rest-api/`
- OpenCitations COCI: `https://opencitations.net/index/coci/api/v1`

## Cached Files

| Source | Cache | Rows | Found |
| --- | --- | ---: | ---: |
| Semantic Scholar | `outputs/external/semantic_scholar_features.csv` | 3047 lookups | 1656 |
| Crossref | `outputs/external/crossref_features.csv` | 1813 DOI lookups | 1813 |
| OpenCitations COCI | `outputs/external/opencitations_features.csv` | 1813 DOI lookups | 1813 |

Semantic Scholar uses both DOI lookups and Semantic Scholar paper ids extracted
from dataset URLs. Crossref and OpenCitations use DOI rows only.

## Features

New feature families:

- Semantic Scholar: `citationCount`, `influentialCitationCount`, `referenceCount`,
  `isOpenAccess`, `publicationTypes`, `fieldsOfStudy`.
- Crossref: `is-referenced-by-count`, `reference-count`, publisher/type/subject,
  license count, abstract availability.
- OpenCitations: DOI citation count from COCI.
- Engineered: best citation count across sources, citation velocity, source
  agreement/disagreement, influential citation ratio, venue/year percentiles.

## Results

Best combined scholarly candidate:

- Submission: `outputs/submissions/next_scholarly_meta_submission.csv`
- Source file: `outputs/scholarly_meta/submissions/huber_scholarly_w0.25_submission.csv`
- Local OOF QWK: `0.602525`
- MAE: `0.855654`
- Distribution: `{1:216, 2:177, 3:107, 4:39, 5:57}`

This is below:

- OpenAlex DOI-only Huber: OOF `0.606685`, public `0.63064`
- Private-safe blend: OOF `0.607028`

## Conclusion

The new sources were crawled successfully, but simply adding all scholarly
features into the meta-model did not improve validation. The likely issue is
noise and source disagreement: citation counts differ across providers, and raw
metadata coverage does not guarantee better ordinal rating predictions.

Do not submit `next_scholarly_meta_submission.csv` before the current better
candidates. Keep it as an analysis artifact.

Recommended next step:

1. Keep `outputs/submissions/next_private_safe_blend_submission.csv` as the next
   Kaggle probe.
2. Use Semantic Scholar influential citations selectively, not as part of a
   large all-source feature dump.
3. Try source-agreement features only where OpenAlex, Semantic Scholar, Crossref,
   and COCI agree on high impact.
