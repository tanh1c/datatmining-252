# Data Mining Next Experiments

## Goal

Current best public score is `0.63064` from the OpenAlex-enriched Huber meta-regressor over the Ridge 5x5 threshold-calibrated title+author anchor. The Kaggle top score around `0.69` suggests that threshold tuning alone is probably not enough; the next gains likely require stronger data mining signals.

## Experiment 1: Metadata + KNN Meta-Signals

Signals tested:

- Ridge 5x5 continuous OOF score as anchor.
- OOF target encoding for `venue`, `year`, `venue_year`, and author surname.
- Title+author TF-IDF KNN label average.

Best local candidate:

- Candidate: `dm_meta_01_combo_vy0.06_auth0.06_knn0.06_submission.csv`
- Local OOF QWK: `0.601942`
- Test distribution: `{1: 217, 2: 172, 3: 116, 4: 33, 5: 58}`

Interpretation: this improves local OOF over plain Ridge threshold, but it changes the test distribution quite a lot and reduces label `4`. It is useful evidence, but not the safest next submission.

## Experiment 2: OpenAlex DOI Enrichment

OpenAlex features were fetched for DOI rows and saved to:

- `outputs/external/openalex_doi_features.csv`

Coverage:

- Train OpenAlex coverage: `58.34%`
- Test OpenAlex coverage: `60.23%`
- DOI rows found in OpenAlex: `1810 / 1813`

Raw citation signals were weak:

- `fwci` vs label correlation: about `+0.020`
- `log_cited_by_count` vs label correlation: about `-0.052`
- `log_referenced_works_count` vs label correlation: about `-0.095`

This means citation count should not be used as a naive monotonic quality score. It is more useful as a meta-feature combined with venue/year/title/author signals.

Best local OpenAlex candidate:

- Candidate: `openalex_huber_meta_w1.00_submission.csv`
- Public LB: `0.63064`
- Local OOF QWK: `0.606685`
- Local MAE: `0.840016`
- Test distribution: `{1: 234, 2: 154, 3: 111, 4: 46, 5: 51}`
- Path: `outputs/openalex_meta/submissions/openalex_huber_meta_w1.00_submission.csv`

Compared with the current `0.62820` anchor, this candidate changes `96` rows:

- Public-test changed rows: `52`
- Private-test changed rows: `44`
- Main transitions: `2->3`, `1->2`, `2->1`, plus smaller changes around `3/4/5`

## Recommendation

The OpenAlex probe was submitted and improved public LB to `0.63064`.

The next Kaggle probes should be:

1. Small blends between `0.63064` OpenAlex Huber and `0.62820` Ridge threshold anchor.
2. OpenAlex title-search enrichment for rows that only have Semantic Scholar URLs.
3. Citation velocity and venue-year-normalized citation percentiles.

Reason: this candidate has the strongest local OOF so far and has now improved public LB, so external scholarly metadata is validated. The next goal is to improve coverage and reduce private-LB risk.

## Next High-ROI Direction

To approach `0.69`, focus on external scholarly metadata and robust validation:

- Expand non-DOI rows by title search through OpenAlex or Semantic Scholar if rate limits allow.
- Add citation velocity features, e.g. citations per year since publication.
- Add venue-year percentile features rather than raw citation count.
- Use a meta-model with OOF-only target encodings and threshold tuning.
- Keep Ridge title+author as the anchor; do not replace it unless the new model wins both OOF and public LB.
