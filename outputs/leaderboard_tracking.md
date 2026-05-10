# Public Leaderboard Tracking

Kaggle public leaderboard is calculated on approximately 50% of the test data.
Final ranking will use the other 50%, so public score is useful feedback but
should not be treated as the final objective.

## Submitted Results

| Public score | Submission | Model | Local CV QWK | Note |
| --- | --- | --- | ---: | --- |
| 0.46096 | `outputs/submissions/randomforest_submission.csv` | RandomForest word TF-IDF | 0.458850 | First valid Stage 2 baseline. CV and public were close. |
| 0.46422 | `outputs/submissions/best_cv_submission.csv` | `blend_top2` = `logreg_word_char` + `logreg_word` | 0.523929 | Best local CV, but only small public gain. |
| 0.48078 | `outputs/submissions/next_logreg_word_char_submission.csv` | LogisticRegression word+char TF-IDF | 0.521188 | Current best public score. Single model beat the higher-CV blend. |
| 0.47369 | `outputs/submissions/next_blend_top3_submission.csv` | `blend_top3` = `logreg_word_char` + `logreg_word` + `linearsvc_word_char` | 0.511721 | Worse than both Logistic single models on public. |
| 0.47950 | `outputs/submissions/next_logreg_word_submission.csv` | LogisticRegression word TF-IDF | 0.506283 | Almost matches word+char public score despite lower CV. |

## Current Takeaways

1. Linear TF-IDF models are clearly stronger than the RandomForest baseline for
   this dataset.
2. Single LogisticRegression models are currently more reliable on public than
   the small blends. The top two public scores are both single Logistic models.
3. `logreg_word_char` remains the best candidate because it has the best public
   score and the best single-model CV QWK.
4. `logreg_word` is surprisingly close on public, which means word-level title,
   venue, author, and year features carry most of the useful signal. Character
   n-grams help, but not by a huge public margin.
5. Higher local CV does not perfectly rank the top submissions. `blend_top2`
   had the best CV but underperformed both Logistic single models on public.
6. `blend_top3` suggests adding `linearsvc_word_char` diluted the strongest
   Logistic signal instead of improving generalization.

## Practical Direction

Keep `outputs/submissions/next_logreg_word_char_submission.csv` as the current
best public checkpoint.

Next experiments should stay close to LogisticRegression rather than broadening
into more blends:

- Tune `logreg_word_char` around `C`, word/char max features, and char n-gram
  range.
- Try a conservative average between `logreg_word_char` and `logreg_word` only
  if the blend gives most weight to `logreg_word_char`.
- Avoid treating public leaderboard as the sole target because private may
  reward the simpler `logreg_word` or a slightly regularized word+char model.
