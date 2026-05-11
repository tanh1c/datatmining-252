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

Keep `outputs/submissions/next_first_surname_c1_5_submission.csv` as the current best
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
