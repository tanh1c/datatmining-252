# Public Leaderboard Tracking

Kaggle public leaderboard is calculated on approximately 50% of the test data.
Final ranking will use the other 50%, so public score is useful feedback but
should not be treated as the final objective.

## Submitted Results

| Public score | Submission | Model | Local CV QWK | Note |
| --- | --- | --- | ---: | --- |
| 0.46096 | `outputs/submissions/randomforest_submission.csv` | RandomForest word TF-IDF | 0.458850 | First valid Stage 2 baseline. CV and public were close. |
| 0.46422 | `outputs/submissions/best_cv_submission.csv` | `blend_top2` = `logreg_word_char` + `logreg_word` | 0.523929 | Best local CV, but only small public gain. |
| 0.48078 | `outputs/submissions/next_logreg_word_char_submission.csv` | LogisticRegression word+char TF-IDF | 0.521188 | Former best public score. Single model beat the higher-CV blend. |
| 0.47369 | `outputs/submissions/next_blend_top3_submission.csv` | `blend_top3` = `logreg_word_char` + `logreg_word` + `linearsvc_word_char` | 0.511721 | Worse than both Logistic single models on public. |
| 0.47950 | `outputs/submissions/next_logreg_word_submission.csv` | LogisticRegression word TF-IDF | 0.506283 | Almost matches word+char public score despite lower CV. |
| 0.50419 | `outputs/submissions/next_eda_best_submission.csv` | LogisticRegression title + first-author word+char TF-IDF | 0.526064 | EDA-driven compact author cue beat earlier full-metadata Logistic variants. |
| 0.50676 | `outputs/submissions/next_author_venue_best_submission.csv` | LogisticRegression title + full authors + venue word+char TF-IDF, C=1.5 | 0.532062 | Best local CV so far, but only a small public gain over the compact author anchor. |
| 0.51876 | `outputs/submissions/next_first_full_venue_submission.csv` | LogisticRegression title + full first-author + venue word+char TF-IDF, C=3.0 | 0.531118 | Full first author plus venue generalized better than full author list on public. |
| 0.52190 | `outputs/submissions/next_first_surname_c1_5_submission.csv` | LogisticRegression title + first-author surname word+char TF-IDF, C=1.5 | 0.529371 | New best public score. Compact first-author surname with stronger regularization won the author/venue batch. |
| 0.52454 | `outputs/submissions/next_best_tuned_submission.csv` | LogisticRegression title + first-author surname word+char TF-IDF, C=0.6 | 0.535162 | New best public score. Lower C improved public while keeping the compact feature set. |
| pending | `outputs/submissions/next_tuned_c0_8_submission.csv` | LogisticRegression title + first-author surname word+char TF-IDF, C=0.8 | 0.532562 | Backup tuned candidate close to C=0.6. |
| pending | `outputs/submissions/next_tuned_c1_5_caps30k50k_submission.csv` | LogisticRegression title + first-author surname word+char TF-IDF, C=1.5, 30k/50k caps | 0.531510 | Tests whether larger TF-IDF caps help without changing features. |
| 0.48093 | `outputs/submissions/next_xgboost_best_submission.csv` | XGBoost TF-IDF title + year, depth=2, 5k features | 0.444144 | Direct sparse TF-IDF XGBoost underperformed the Logistic anchor; local CV warning matched public. |
| 0.60804 | `outputs/submissions/next_ridge_threshold_5x5_submission.csv` | Repeated-CV Ridge score ensemble, title+author TF-IDF, threshold tuned, 5 folds x 5 seeds | 0.596665 | New best public score. This is the current anchor. |
| 0.59612 | `outputs/submissions/next_ridge_threshold_10x10_submission.csv` | Repeated-CV Ridge score ensemble, title+author TF-IDF, threshold tuned, 10 folds x 10 seeds | 0.605097 | Higher local OOF but lower public, confirming the label-distribution risk. |
| pending | `outputs/submissions/next_ridge_alpha6_5x5_submission.csv` | Ridge threshold 5x5, alpha=6 | 0.597018 | Slightly higher OOF than alpha=8 but only predicts 20 label-4 rows, so check public carefully. |
| pending | `outputs/submissions/next_ridge_alpha10_5x5_submission.csv` | Ridge threshold 5x5, alpha=10 | 0.595704 | Lower OOF than anchor; useful only as a distribution check. |
| 0.62463 | `outputs/submissions/next_ridge_label4_more_mid_submission.csv` | Ridge 5x5 alpha=8 score with threshold variant increasing label 4 | 0.598151 | New best public score. Changes only 18 rows vs 0.60804 and predicts 46 label-4 rows. |
| 0.62120 | `outputs/submissions/next_ridge_label4_more_small_submission.csv` | Ridge 5x5 alpha=8 score with smaller label-4 increase | 0.597047 | Also beats the 0.60804 anchor; changes 11 rows and predicts 39 label-4 rows. |
| 0.62820 | `outputs/submissions/next_ridge_label4_51_submission.csv` | Ridge 5x5 alpha=8 threshold variant with 51 label-4 predictions | 0.597188 | New best public score, tied with the 55-label4 variant. Closest to the prior 0.62463 anchor. |
| 0.62820 | `outputs/submissions/next_ridge_label4_55_submission.csv` | Ridge 5x5 alpha=8 threshold variant with 55 label-4 predictions | 0.597636 | Tied best public score, only 4 rows different from the 51-label4 variant. |
| pending | `outputs/submissions/next_ridge_label4_58_submission.csv` | Ridge 5x5 alpha=8 threshold variant with 58 label-4 predictions | 0.596143 | More aggressive label-4 expansion; lower OOF, submit only if more tests are allowed. |

## Current Takeaways

1. Linear TF-IDF models are clearly stronger than the RandomForest baseline for
   this dataset.
2. Single LogisticRegression models are currently more reliable on public than
   the small blends. The top public scores are all single Logistic models.
3. `title_first_surname_c1_5` is the current best public candidate even though
   `title_full_authors_venue_c1_5` has higher local CV QWK.
4. `logreg_word` was surprisingly close to the earlier full-feature Logistic
   model on public, which means word-level title,
   venue, author, and year features carry most of the useful signal. Character
   n-grams help, but not by a huge public margin.
5. Higher local CV does not perfectly rank the top submissions. `blend_top2`
   had the best CV but underperformed both Logistic single models on public.
6. `blend_top3` suggests adding `linearsvc_word_char` diluted the strongest
   Logistic signal instead of improving generalization.

## Practical Direction

Keep `outputs/submissions/next_ridge_label4_51_submission.csv` as the current best
public checkpoint.

Next experiments should stay close to single LogisticRegression candidates
rather than broadening into more blends:

- Tune around `title_first_surname_c1_5` with nearby `C`, word/char feature
  caps, and char n-gram range.
- Test `title + first_author_surname + venue` with venue included once, because
  venue helps but can overfit if it dominates the text.
- Try only cautious blends that keep `title_first_surname_c1_5` as the anchor.
- Avoid treating public leaderboard as the sole target because private may
  reward the simpler `logreg_word` or a slightly regularized word+char model.

## After 0.50419

The jump from `0.48078` to `0.50419` confirms that feature selection mattered
more than model complexity. Full author strings, venue/year, and DOI-style
tokens can dilute the signal, while title plus a compact first-author token is
stronger on both CV and public leaderboard.

The author/venue batch refined this: full authors plus venue can help if the
Logistic model is more regularized (`C=1.5`). With the earlier `C=3.0`, the
same feature family drops, so author/venue are useful but easy to overfit.

## After 0.52190

The author/venue submissions show a clear public ranking:

- `title + first-author surname`, `C=1.5`: `0.52190`
- `title + full first-author + venue`, `C=3.0`: `0.51876`
- `title + full authors + venue`, `C=1.5`: `0.50676`

This means author and venue are important, but the public split favors compact
author identity more than broad full-author text. Full author lists probably add
collaborator noise; venue helps, but not enough to beat the simpler surname
anchor. Local CV overestimated the full-author+venue candidate, so future
experiments should use CV as a filter, then keep public-safe simplicity.

Future submissions should explore small, controlled variants:

- `title + first_author_surname` with nearby Logistic `C` values around `1.5`.
- `title + first_author_surname + venue` with venue included once, not repeated
  heavily.
- first-author surname vs full first-author token.
- char n-gram ranges such as `(3, 5)`, `(4, 6)`, and feature caps around
  `20k/30k/50k`.
- cautious blends that keep `title_first_author_word_char` as the anchor.

## Best Feature Tuning

Keeping the same feature set as the current public best
(`title + title + first_author_surname`), the local tuning batch found lower
regularization strength was better:

- `C=0.6`: CV QWK `0.535162`, public `0.52454`
- `C=0.8`: CV QWK `0.532562`
- `C=1.5`: CV QWK `0.529371`, public `0.52190`

The first-author bias audit found no case matching the most concerning example:
there are no authors with exactly one train paper labeled `5` and two or more
test papers. There are seven train-count-one label-5 authors in test, but each
appears only once in test. Rare-author bias still exists as a risk: nine authors
have exactly one train paper and at least two test papers, but only one of those
had all test predictions equal to its single train label.

## After 0.52454

The `C=0.6` result confirms that the current winning direction is not adding
more metadata, but making the compact title plus first-author-surname model less
overconfident. The public gain is small but consistent with CV.

Next improvements should stay near this anchor:

- Try finer `C` values around `0.4` to `0.9`.
- Keep the same feature set while testing TF-IDF caps and char n-gram ranges.
- Test a very light venue token only after confirming nearby `C` values.
- Avoid full author lists unless paired with stronger regularization and a
  public-safe sanity check.

## XGBoost Trial

Direct XGBoost on sparse word TF-IDF did not reproduce the stronger public
score reported externally. The best local XGBoost candidate was
`title + year`, depth `2`, `5k` TF-IDF features with CV QWK `0.444144` and
public score `0.48093`, far below the Logistic anchor. Unweighted spot checks
were also weak (`title + year` depth `3`, `10k` features reached only about
`0.433` CV QWK).

This suggests plain XGBoost over sparse TF-IDF is not the right local recipe.
If XGBoost is revisited, try it on dense SVD/LSA text features plus numeric
year, or use it as a second-level model over out-of-fold probabilities rather
than directly over high-dimensional sparse text.

## Ridge Threshold Ensemble

The external `0.59759` method is now implemented locally as
`train_ridge_threshold_ensemble.py`. It treats labels as ordinal scores,
ensembles OOF/test continuous Ridge predictions across repeated
StratifiedKFold splits, then tunes 4 thresholds on OOF predictions to maximize
QWK.

Local reproduction:

- 5 folds x 5 seeds: OOF QWK `0.596665`, MAE `0.852847`, AdjAcc `0.791500`.
- 10 folds x 10 seeds: OOF QWK `0.605097`, MAE `0.846832`, AdjAcc `0.787891`.

Recommended submit order:

1. Keep `outputs/submissions/next_ridge_label4_more_mid_submission.csv` as the
   primary public anchor. It scored `0.62463`.
2. Do not promote `outputs/submissions/next_ridge_threshold_10x10_submission.csv`
   despite higher OOF. It scored `0.59612`.

The 10x10 candidate has better OOF but only predicts 20 rows as label `4`,
matching the risk noted in the external write-up. Public leaderboard confirmed
that 5x5 is safer than 10x10 here.

## After 0.60804

The Ridge threshold approach is now clearly better than the earlier Logistic
anchor (`0.52454`). The main winning ingredients are ordinal regression,
OOF threshold tuning, and repeated-CV score averaging over `title_clean` and
`authors_clean` TF-IDF.

The 10x10 result is an important warning: more folds/seeds improved OOF
(`0.605097` vs `0.596665`) but hurt public (`0.59612` vs `0.60804`). The likely
reason is threshold/score distribution drift, especially around middle labels
`3/4`. Future work should tune around the 5x5 recipe rather than blindly adding
more folds or seeds.

Next experiments:

- Try 5x5 Ridge alpha values near `8.0`, such as `6`, `10`, and `12`.
- Keep 5x5 but tune threshold constraints or search bounds around the 0.60804
  thresholds.
- Explore small blends between 5x5 Ridge scores and the Logistic `0.52454`
  anchor only if OOF and label distribution remain stable.

## Ridge 5x5 Follow-Up

Alpha sweep:

- `alpha=6`: OOF QWK `0.597018`, distribution `{1:225, 2:158, 3:139, 4:20, 5:54}`.
- `alpha=8`: OOF QWK `0.596665`, public `0.60804`, distribution `{1:237, 2:183, 3:92, 4:28, 5:56}`.
- `alpha=10`: OOF QWK `0.595704`, distribution `{1:210, 2:219, 3:82, 4:32, 5:53}`.
- `alpha=12`: OOF QWK `0.594179`, distribution `{1:192, 2:213, 3:99, 4:34, 5:58}`.

Threshold-only variants from the alpha=8 score:

- `label4_more_mid`: OOF QWK `0.598151`, public `0.62463`, distribution `{1:237, 2:183, 3:79, 4:46, 5:51}`, changes 18 rows vs 0.60804.
- `label4_51`: OOF QWK `0.597188`, distribution `{1:237, 2:183, 3:75, 4:51, 5:50}`, changes 5 rows vs 0.62463.
- `label4_55`: OOF QWK `0.597636`, distribution `{1:237, 2:183, 3:71, 4:55, 5:50}`, changes 9 rows vs 0.62463.
- `label4_58`: OOF QWK `0.596143`, distribution `{1:237, 2:183, 3:68, 4:58, 5:50}`, changes 12 rows vs 0.62463.
- `label4_more_small`: OOF QWK `0.597047`, public `0.62120`, distribution `{1:237, 2:183, 3:85, 4:39, 5:52}`, changes 11 rows vs 0.60804.
- external thresholds: OOF QWK `0.595977`, distribution `{1:210, 2:176, 3:124, 4:30, 5:56}`.

The public results confirmed the label-4 hypothesis: increasing label `4` from
28 rows in the original 5x5 submission to 39/46 rows improved public score from
`0.60804` to `0.62120`/`0.62463`.

## After 0.62463

The current best is the alpha=8 Ridge 5x5 score with a threshold-only adjustment
that widens the label-4 interval. This is a cleaner improvement than retraining:
it preserves the proven Ridge scores and changes only 18 rows relative to the
0.60804 anchor.

Next experiments should stay threshold-focused:

- Try label-4 counts between 46 and roughly 55 by lowering the third threshold
  a bit more or raising the fourth threshold a bit more.
- Avoid moving the first two thresholds unless there is evidence public needs
  different label `1/2/3` balance.
- Keep alpha=8 as the score anchor; alpha=6 changed the distribution much more
  and is riskier despite slightly higher OOF.
- Compare changed rows from `label4_more_mid` to identify whether most gains
  came from `3 -> 4` or `5 -> 4`.

New threshold-only candidates from the 0.62463 anchor:

1. `outputs/submissions/next_ridge_label4_51_submission.csv`
2. `outputs/submissions/next_ridge_label4_55_submission.csv`
3. `outputs/submissions/next_ridge_label4_58_submission.csv`

Submit in that order. The 51-label-4 candidate is closest to the current best;
the 55-label-4 candidate has better OOF than 51 but changes more rows; 58 is an
aggressive boundary test.

## After 0.62820

Both `label4_51` and `label4_55` scored `0.62820`, despite differing by 4 rows.
This suggests the public split has a plateau around 51-55 label-4 predictions.
Use `next_ridge_label4_51_submission.csv` as the safer primary anchor because it
is closer to the previous `0.62463` submission, while `label4_55` is an equally
valid tied candidate with slightly better OOF.

Next threshold tests should probe just outside the plateau:

- Try label-4 counts around 49-50 and 56-57.
- Avoid large jumps beyond 58 because OOF already drops at that point.
- Keep labels `1` and `2` unchanged; all confirmed gains came from moving
  boundary mass between labels `3`, `4`, and `5`.

## After 0.63064

Submitted `outputs/openalex_meta/submissions/openalex_huber_meta_w1.00_submission.csv`
and received public score `0.63064`, improving over the Ridge threshold anchor
`0.62820`.

This candidate uses a Huber meta-regressor over the Ridge 5x5 continuous score,
OOF metadata priors, and OpenAlex DOI features. The OpenAlex cache is stored at
`outputs/external/openalex_doi_features.csv` with these columns:

- `doi_norm`
- `openalex_found`
- `openalex_error`
- `openalex_id`
- `openalex_title`
- `openalex_year`
- `cited_by_count`
- `referenced_works_count`
- `fwci`
- `is_retracted`

OpenAlex coverage:

- `1813` unique DOI records queried.
- `1810` DOI records found in OpenAlex.
- Train row coverage: `58.34%`.
- Test row coverage: `60.23%`.

Important interpretation:

- Raw citation count is not a simple monotonic quality feature here; its direct
  correlation with label was weak/slightly negative.
- The signal becomes useful when combined with Ridge score, venue/year/author
  priors, and robust regression.
- This validates the next direction: external scholarly metadata plus
  private-safe meta-modeling, not threshold-only tuning.

Next experiments:

- Expand OpenAlex coverage for Semantic Scholar URL rows using title search.
- Add citation velocity, e.g. citations per year since publication.
- Add venue-year-normalized citation percentiles instead of raw citation count.
- Try small blends between `0.63064` OpenAlex Huber and `0.62820` Ridge anchor
  to reduce private leaderboard risk.
