# 0.63064 Score Archive

Public score: `0.63064`

Submission:

- `openalex_huber_meta_w1.00_submission.csv`

Method:

- Huber meta-regressor using the Ridge 5x5 score anchor plus metadata priors and OpenAlex DOI features.
- Threshold tuning on OOF predictions for QWK.

OpenAlex data:

- Source cache: `openalex_doi_features.csv`
- Rows: `1813` unique DOI records
- Found in OpenAlex: `1810`
- Columns: `doi_norm, openalex_found, openalex_error, openalex_id, openalex_title, openalex_year, cited_by_count, referenced_works_count, fwci, is_retracted`

Submission validation:

- Rows: `596`
- Label distribution: `{1: 234, 2: 154, 3: 111, 4: 46, 5: 51}`

Conclusion:

Public score improved from `0.62820` to `0.63064`, so external scholarly metadata is now validated as a useful direction. Continue with OpenAlex/title-search enrichment and private-safe blends around this candidate.
