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
| **0.69972** | `outputs/specter2_finetune/specter2_finetune_submission.csv` | Fine-tuned SPECTER2 base + regression head, 5 folds x 3 seeds, threshold tuning | 0.638866 | First major jump to 0.70. +0.069 over previous. Semantic encoder of `title + abstract` with fine-tuning unlocks the topic-relevance axis. |
| 0.68737 | `outputs/specter2_finetune_5seed/specter2_finetune_submission.csv` | Same notebook, **5 seeds** instead of 3 (`[252,253,254,255,256]`), 25 models | 0.644537 | OOF higher than 0.69972 but public **lower by 0.012**. Threshold drift inflated predicted label 2 to 36% vs train 21%. See `outputs/lessons_learned.md` (L1). |
| 0.68718 | `outputs/0.68718/specter2_finetune_v2_submission.csv` | Step 3c: v3 abstracts (95.7%) + max_len 384 + constrained tuner (lambda=0.5). 5 folds x 3 seeds | 0.644552 | Same regression pattern as 5-seed: OOF +0.006, public -0.012. Three knobs changed at once. See `outputs/lessons_learned.md` (L6). |
| **0.71052** | `outputs/0.71052/blend_2anchor_70_specter_submission.csv` | Stacking step 5 — `0.7 * specter_3s_oof + 0.3 * ridge_5x5_oof`, distribution-constrained thresholds | 0.646351 | First submission to break 0.70. +0.011 over 0.69972. 2-anchor minimalist blend (1 semantic + 1 lexical) beat 4-anchor stacks. |
| **0.72103** | `outputs/0.72103/blend_3anchor_scincl_specter_ridge_60_20_20_submission.csv` | **Stacking step 6 — `0.6 * scincl + 0.2 * specter_3s + 0.2 * ridge_5x5`, distribution-constrained thresholds** | **0.641159** | **NEW BEST — first submission past 0.72. +0.011 over 0.71052. SciNCL added genuine signal despite Pearson 0.943 correlation with SPECTER2 and slightly lower OOF (0.6269). Both heuristics (OOF QWK and test L1) said "do not submit"; submitted anyway as a hedge with the lowest-test-L1 SciNCL candidate. See L9 in lessons_learned.md.** |
| 0.71622 | `outputs/0.71622/blend_3anchor_scincl_specter_ridge_70_20_10_submission.csv` | Sweep around 60/20/20 — 70% scincl + 20% specter + 10% ridge | 0.639286 | Regression vs 0.72103 (−0.005). Lower test L1 (0.158) and slightly lower OOF (-0.002) both said "might transfer", but reducing Ridge from 20% to 10% hurt. See L10 in lessons_learned.md. |
| **0.72374** | `outputs/submissions/next_bge_4anchor_safest_submission.csv` | **Step 10 — 4-anchor stack `bge 0.40 / scincl 0.40 / specter 0.15 / ridge 0.05`, distribution-constrained thresholds** | **0.6567** | **NEW BEST — first submission past 0.72103. +0.00271 over previous best. BGE-large-en-v1.5 fine-tune cleared both L12 floors (signal 0.6228, r 0.909 vs SPECTER2). Picked by L7/L9: lowest test_L1 (0.1556) among candidates with OOF > anchor + 0.005. OOF lift +0.0155 translated to only +0.003 public — see L13 in lessons_learned.md. |
| 0.70965 | `outputs/submissions/next_bge_4anchor_single_knob_submission.csv` | Step 10 follow-up — `bge 0.30 / scincl 0.40 / specter 0.10 / ridge 0.20` | 0.6572 | **Regression** vs `safest` (−0.01409). Multi-knob change (ridge 0.05→0.20, bge 0.40→0.30, specter 0.15→0.10) — primary culprit is ridge ↑0.15 since BGE already carries lexical-adjacent signal (r 0.818 vs ridge), making heavy ridge double-counting. **L14:** public-validated weights are stack-configuration-specific; L10's "ridge=0.20 safe" rule was for the 3-anchor SciNCL/SPECTER/Ridge stack, does NOT transfer when BGE is added. |
| **0.72394** | `outputs/submissions/next_e5_4anchor_safest_e5_submission.csv` | **Step 13 — E5 4-anchor stack `e5 0.50 / scincl 0.30 / specter 0.20 / ridge 0.00`, distribution-constrained thresholds** | **0.6593** | **NEW BEST — +0.00020 over BGE safest (0.72374). E5-large-v2 OOF 0.6417 (highest single-anchor) but r 0.944 vs BGE = same-family redundancy (L7 boundary). Lift was tiny because E5 = "better BGE" not "new anchor". Picked by L7/L9: lowest test_L1 0.1422 in any sweep (vs anchor 0.156). Step 13 confirmed E5+BGE 5-anchor stacks always underperform 4-anchor with E5 replacing BGE. See L15 in lessons_learned.md. |
| not submitted | `outputs/deberta_v3_finetune_outputs/deberta_v3_finetune/` | DeBERTa-v3-large fine-tune (435M, mean pool + LLRD 0.95, fp32) | 0.568894 | **Dropped, never submitted.** Round-QWK 0.5511 vs SPECTER2 0.6297 / SciNCL 0.6117. Pearson r 0.848 vs SPECTER2 — diverse but weak signal. Adding to the 60/20/20 anchor at any weight 5-20% **regressed** OOF round-QWK from 0.6074 to 0.5978. See L11 in lessons_learned.md. |
| not submitted | `outputs/openalex_anchor/` | OpenAlex non-text anchor (Huber over DOI+title-search citation features + venue/year/author target encoding, 5x3) | 0.364627 | **Dropped, never submitted.** Round-QWK 0.2570 (signal too weak) but Pearson r only 0.499 vs SPECTER2 (most diverse candidate ever). Best blend `ridge -0.05 +openalex 0.05` lifts round-QWK by +0.0008 — within noise band. Citation features track impact, not ASP relevance. See L12 in lessons_learned.md. |
| not submitted | `outputs/e5_large_finetune_max384_outputs/` | E5-large-v2 ablation MAX_LEN 256→384 (single-knob audit) | 0.6433 | **Ablation only, not for stacking.** Tested whether the friend's SPECTER2 recipe winning trick (max_len=384) transfers to E5. **It does not.** OOF dropped −0.0063 vs step12 E5 (0.6496). Confirms contrastive-sentence encoders (E5/BGE/MS-MARCO siblings) prefer max_len=256 because their pre-training pairs average ~150 tokens; 384 is OOD. SPECTER2 trained on full abstracts so 384 fits there. **L16:** mindset bias cleared — max_len=256 was correct for E5/BGE; freeze rule from L6 needs a re-test escape valve. |


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

## OpenAlex Title Search + Blend Follow-Up

Implemented `train_openalex_title_meta.py` to make the external-feature path
reproducible. The script expands OpenAlex features with title search for rows
without DOI, adds citation velocity and venue/year percentile features, then
trains Huber/Ridge meta candidates.

Coverage after title search:

- Train: `1455` DOI matches, `742` title matches, `297` no OpenAlex match.
- Public test: `177` DOI matches, `100` title matches, `21` no match.
- Private test: `182` DOI matches, `93` title matches, `23` no match.

Best title-search candidate:

- `outputs/submissions/next_openalex_title_meta_submission.csv`
- Local OOF QWK `0.603484`
- Distribution `{1:252, 2:124, 3:119, 4:45, 5:56}`

Best private-safe blend:

- `outputs/submissions/next_private_safe_blend_submission.csv`
- Blend: `65%` OpenAlex DOI-only Huber + `35%` Ridge `0.62820` anchor.
- Local OOF QWK `0.607028`
- Distribution `{1:198, 2:197, 3:114, 4:32, 5:55}`

Submit `next_private_safe_blend_submission.csv` first. It has the strongest OOF
so far and directly tests whether a conservative blend between the two
public-proven anchors improves public LB. Submit the title-search candidate only
as a second probe because its OOF is lower than the DOI-only OpenAlex model.

## Scholarly Sources Follow-Up

Implemented `train_scholarly_meta.py` to add Semantic Scholar, Crossref, and
OpenCitations COCI features on top of the OpenAlex caches.

New caches:

- `outputs/external/semantic_scholar_features.csv`: `3047` lookups, `1656` found.
- `outputs/external/crossref_features.csv`: `1813` DOI lookups, all found.
- `outputs/external/opencitations_features.csv`: `1813` DOI lookups, all found.

Best combined candidate:

- `outputs/submissions/next_scholarly_meta_submission.csv`
- Local OOF QWK `0.602525`
- Distribution `{1:216, 2:177, 3:107, 4:39, 5:57}`

This is lower than the OpenAlex DOI-only Huber candidate (`0.606685`) and the
private-safe blend (`0.607028`). Conclusion: the sources were crawled
successfully, but an all-source feature dump adds noise. Do not submit this
candidate before `next_private_safe_blend_submission.csv`.

## Scholarly Feature Selection

Ran `analyze_scholarly_feature_selection.py` to reduce noise from the new
external-source features. The key result is that compact feature groups are
better than the all-source dump.

Best group:

- `opencitations_core`
- Features: `coci_found`, `log_coci_citation_count`
- Model: Huber meta-model blended with Ridge anchor
- Best blend: `65%` filtered external model + `35%` Ridge anchor
- Local OOF QWK: `0.608147`
- Distribution: `{1:216, 2:176, 3:118, 4:35, 5:51}`
- Submission: `outputs/submissions/next_filtered_scholarly_submission.csv`

Feature-family ranking by OOF:

1. `opencitations_core`: `0.608147`
2. `source_agreement`: `0.607076`
3. `base_no_external`: `0.605850`
4. `openalex_core`: `0.605720`
5. `crossref_core`: `0.604444`

Conclusion: for now, keep **COCI citation count** as the cleanest new external
signal. Semantic Scholar and Crossref contain useful-looking fields, but adding
them wholesale is noisy. Submit `next_filtered_scholarly_submission.csv` before
the broader `next_scholarly_meta_submission.csv`.

## After Scholarly Public Scores

Two scholarly candidates were submitted:

- `outputs/submissions/next_scholarly_meta_submission.csv`: public `0.60531`.
- `outputs/submissions/next_filtered_scholarly_submission.csv`: public `0.61227`.

Both are far below the current best `0.63064`.

Interpretation:

- The all-source scholarly dump is confirmed bad on public LB.
- The filtered COCI candidate had the best local OOF (`0.608147`) but only
  reached public `0.61227`, so its OOF gain was not leaderboard-stable.
- This is a strong warning that external citation-style features are very noisy
  for the public split unless they are used exactly as in the proven OpenAlex
  DOI-only Huber candidate.

Decision:

- Keep `0.63064` OpenAlex DOI-only Huber as the best public anchor.
- Do not submit more COCI/Crossref/Semantic Scholar variants without a new
  validation idea.
- Future improvements should focus on understanding why the `0.63064` OpenAlex
  model works, not on adding more raw impact sources.



## After 0.69972 — SPECTER2 Fine-tune (NEW BEST)

A fine-tuned SPECTER2 encoder on `title + abstract` pushed public LB from
`0.63064` to **`0.69972`**, a `+0.069` jump — the largest in the project so far.

The full recipe is archived in `outputs/0.69972/manifest.md`. Highlights:

- Step 1: confirmed the hypothesis that Label = ASP/AI-symbolic topic relevance
  (rule-based proxy QWK 0.2589 from 5 hand-written keyword groups, train-mean
  Label `3.92` vs `2.29` for ASP-keyword vs others).
- Step 2: multi-source abstract crawl (Semantic Scholar, OpenAlex, Crossref,
  OpenAlex title-search) -> `outputs/external/abstracts_merged_v2.csv` covering
  93.7% of train+test rows.
- Step 3b: fine-tuned `allenai/specter2_base` with a `Linear(768->1)` head,
  SmoothL1 loss on the float Label, AdamW with split LR (encoder `2e-5`, head
  `1e-3`), 5 epochs, mixed precision, 5 folds x 3 seeds = 15 models, OOF
  threshold tuning. Notebook: `notebooks/step3b_specter2_finetune.ipynb`.
- OOF QWK `0.6389`, label distribution `{1:252, 2:128, 3:102, 4:63, 5:51}`,
  public/private label-5 split 26/25 (balanced).

Why it works: SPECTER2 base already encodes scientific paper similarity, so
fine-tuning re-shapes that representation around the *label* axis with very
little data. TF-IDF Ridge plateaued at 0.628 because it cannot semantically
unify "ASP" / "answer set programming" / "stable model"; SPECTER2 does.

The frozen-SPECTER2 attempt (step 3a) only reached 0.49 OOF and predicted just
14 label-5 rows out of 596 — fine-tuning is what makes the difference.

### Next directions

1. **Stacking (Step 5)**: blend `0.69972` SPECTER2-finetune OOF with Ridge
   `0.62820` OOF and OpenAlex `0.63064` OOF. Three near-uncorrelated signals
   (lexical, semantic, citation) usually add +0.01-0.03.
2. **5-seed rerun** of the same notebook (~50 min on A100): fold log shows seed
   `254` gave higher per-fold QWK (`0.64-0.66`); two more seeds should push OOF
   to `~0.645-0.65`.
3. **LLM scoring (Step 4)**: zero/few-shot ASP-relevance label as an extra
   meta-feature.
4. **Token length sweep**: 256 -> 384 / 512 max_len. Some abstracts are 1500+
   chars, so longer context may help.
5. **SPECTER2 augmented base** (`allenai/specter2_aug2023refresh_base`).


## After 0.68737 — 5-seed regression confirms the threshold-drift trap

A 5-seed rerun of the exact same notebook (only `SEEDS` extended from
`[252, 253, 254]` to `[252, 253, 254, 255, 256]`) raised OOF QWK from
`0.6389` to `0.6445` (+0.006) but lowered public LB from `0.69972` to
`0.68737` (−0.012).

Threshold 1 dropped from `1.808` to `1.597`, which moved 72 borderline papers
from label 1 to label 2. Predicted public split distribution went from
`{1:39%, 2:22%, 3:20%, 4:11%, 5:9%}` (close to train) to `{1:29%, 2:36%,
3:16%, 4:10%, 5:9%}` (label 2 inflated to ~2x its expected share). The
`differential_evolution` optimiser is greedy on OOF QWK and does not penalise
distribution drift.

This is the **second** time we hit this pattern (the first was 5x5 Ridge
`0.60804` vs 10x10 Ridge `0.59612` last week). The rule is now codified in
`outputs/lessons_learned.md` (L1, L4).

**Action items folded into next steps:**

1. Keep `0.69972` as the SPECTER2 anchor for stacking. Use the `0.68737` OOF
   + test scores as a *diversity feature* alongside it.
2. Step 6 must implement a *distribution-constrained* threshold tuner that
   penalises `|pred_share_k - train_share_k|`.
3. When a future submission gives higher OOF but a shifted distribution, do
   not promote it; retune thresholds with the constraint first.


## After 0.71052 — First submission past 0.70 (stacking blend)

A 2-anchor blend `0.7 * specter_3s_oof + 0.3 * ridge_5x5_oof` with
distribution-constrained threshold tuner pushed public LB from `0.69972` to
**`0.71052`**, a `+0.0108` jump. Full recipe in `outputs/0.71052/manifest.md`.

Twelve stack candidates were tried. The candidate with the **highest OOF
QWK** (`huber_meta`, 0.6534) was *not* the one that won — it dropped the
specter_3s anchor (the public-best base) because the 3 SPECTER2 anchors are
~0.98 correlated and meta-models tend to drop one of them. Instead, the
candidate with the **lowest test combined L1 distance** (0.152) won — it had
OOF 0.6464 (lower than huber_meta) but its predicted label distribution was
closest to the train distribution.

This is L7 in `outputs/lessons_learned.md`: when stacking, rank candidates by
test L1 distance first and OOF QWK second.

### Where to go next

| Candidate | Why | Expected gain |
| --- | --- | --- |
| Fine-tune **SciBERT** as a 4th truly diverse anchor | SciBERT vs SPECTER2 should correlate <0.85, unlike specter_5s/v2 which were ~0.98 vs specter_3s | +0.005 - 0.012 |
| Re-run OpenAlex script to extract continuous OOF and add as 5th anchor | citation-based signal is uncorrelated with text | +0.002 - 0.008 |
| LLM zero/few-shot ASP-relevance score | another orthogonal signal direction | +0.005 - 0.015 |
| Sweep blend weight 0.65/0.35, 0.75/0.25 | cheap hyperparameter explore | +0.000 - 0.005 |

The stacking ceiling is bounded by the **diversity** of base anchors. We
already have 2 effective signals (semantic + lexical). To push further we
need a genuinely third signal, not another SPECTER2 variant.


## After 0.72103 — Three-anchor stack with SciNCL added (NEW BEST, breaks 0.72)

A 3-anchor weighted blend `0.6 * scincl + 0.2 * specter_3s + 0.2 * ridge` with
distribution-constrained threshold tuner pushed public LB from `0.71052` to
**`0.72103`**. Full recipe in `outputs/0.72103/manifest.md`.

This submission was counter-intuitive on paper:

- OOF QWK 0.6412 (lower than the 0.71052 anchor's 0.6464).
- Test combined L1 distance 0.162 (higher than the anchor's 0.152).
- SciNCL alone OOF 0.6269 (lower than every other base anchor).
- SciNCL Pearson correlation 0.943 with SPECTER2 (similar to the SciBERT
  correlation that did not help).

It still works because high pairwise correlation between two strong base
models is **not** the same as redundancy. SciNCL was trained with a different
contrastive objective (neighborhood contrastive learning vs SPECTER2's
triplet loss) so its disagreement set with SPECTER2 falls on different rows
than where SciBERT disagreed.

**This is L9 in `outputs/lessons_learned.md`:** treat OOF gaps < 0.01 and
test-L1 gaps < 0.03 as noise relative to the anchor. Submit borderline
candidates one at a time; do not over-prune by metric heuristics alone.

### Where to go next

| Idea | Why | Expected gain |
| --- | --- | ---: |
| Sweep weights around 60/20/20: 55/25/20, 65/15/20, 70/10/20 | Confirm winning region; cheap | +0.000 → +0.005 |
| Add a 4th architecturally-different anchor (DeBERTa-v3-large, bge-large) | More diverse signals | +0.005 → +0.012 |
| Try SPECTER2 `classification` adapter as a 5th anchor | Different head, same encoder | +0.000 → +0.005 |
| Use 35/35/30 weights (highest OOF among 3-anchor SciNCL blends) | Test whether higher OOF transfers despite higher test L1 | +0.000 → +0.010 (untested) |

Top priority is the weight sweep around 60/20/20 because it costs nothing
extra and tests a concrete hypothesis.


## After DeBERTa-v3-large — Drop step 8, generic encoder under-fits scientific text

A 5×3 fine-tune of `microsoft/deberta-v3-large` was attempted as the 4th
anchor (step 8 / `notebooks/step8_deberta_v3_finetune.ipynb`). It was
**not submitted**. Full evidence in `outputs/lessons_learned.md` (L11).

Why it died:

- OOF round-QWK **0.5511** (constrained-tuned 0.5689). Below SPECTER2
  (0.6297) and SciNCL (0.6117) on the same OOF rows. Above the L8 floor
  (~0.51) but only marginally.
- Pearson r vs SPECTER2 = 0.848, vs SciNCL = 0.859. Diversity is real
  (lower than the 0.943 SPECTER2 / SciNCL share) but the signal is too
  weak to convert that diversity into stack lift.
- **Direct blend probe (no threshold tune):** the 60/20/20 anchor scores
  round-QWK 0.6074. Adding DeBERTa-v3 at any weight from 5% to 20%
  monotonically regresses to 0.6037 / 0.5997 / 0.5983 / 0.5978. The
  candidate is dead weight.

Tuning timeline (all standard DeBERTa-v3 fine-tune fixes were applied):

| Variant | best per-fold |
| --- | ---: |
| CLS pool, fp16/bf16 autocast | NaN loss (disentangled attention overflow) |
| CLS pool, fp32, LR_enc 1e-5, 5 epochs | 0.502 |
| CLS pool, fp32, LR_enc 2e-5, 6 epochs | 0.533 |
| Mean pool, fp32, LR_enc 2e-5, 6 epochs | 0.570 |
| Mean pool + LLRD 0.95, 5 epochs | 0.580 |

Root cause: DeBERTa-v3 was pre-trained on CommonCrawl + Wikipedia +
Books — generic English with no scientific-paper inductive bias.
SPECTER2 / SciNCL / SciBERT were pre-trained with citation-contrastive
objectives on scientific abstracts, so they start with a representation
already aligned with the label axis.

This matches L8 (LLM zero-shot drop) almost exactly: a candidate with
better diversity but weaker signal than the existing anchors does not
help. Diversity only matters above an OOF floor of ~80% of the strongest
anchor's OOF (~0.51 here). DeBERTa-v3 cleared that floor by 0.04 — not
enough.

Updated direction for the 4th anchor (replacing the row in the table
above):

| Idea | Why | Status |
| --- | --- | --- |
| ~~DeBERTa-v3-large~~ | ~~more diverse signal~~ | **Dropped (L11)** — generic CC pretraining mismatched scientific text |
| ~~OpenAlex citation OOF as continuous anchor~~ | ~~non-text signal; uncorrelated with all text encoders~~ | **Dropped (L12)** — most diverse candidate ever (r 0.499) but signal too weak (OOF 0.257); citations track impact, not ASP relevance |
| `bge-large-en-v1.5` or `e5-large-v2` | also generic-large, but pre-trained with **contrastive sentence similarity** — closer to SPECTER2's objective, signal more likely to transfer | **DONE — public 0.72374** (step 9 + step 10), see "After 0.72374" below |
| Venue / first-author target encoding | non-text signal; cheap | not yet tried |
| Qwen2.5-7B / Llama-3.1-8B LoRA fine-tune | LLM zero-shot lacked signal (L8); fine-tune may work where DeBERTa-v3 didn't because reasoning span differs from CLS pooling | bigger bet (3-4h GPU) |

Practical rule added to L11: **before committing to a full 15-fold run on
a new anchor, run a quick "+10% blend probe" on its OOF.** If the round-QWK
of `(60/20/20 anchor) + 0.10 * candidate` is below the anchor's
round-QWK, abort the run.


## After OpenAlex anchor — Drop, signal-floor failure (L12)

A 5×3 OpenAlex continuous anchor was built (`src/build_openalex_anchor.py`,
`outputs/openalex_anchor/`) using Huber regression over OpenAlex DOI +
title-search citation features (cited_by_count, FWCI, citation velocity,
venue/year percentiles), citation-age, and OOF target encodings on
venue/year/first-surname. It was **not submitted**. Full evidence in
`outputs/lessons_learned.md` (L12).

Why it died:

- OOF round-QWK **0.2570** (constrained-tuned 0.3646). This is below
  every other anchor and well under the L8 signal floor (~0.51).
- Pearson r vs SPECTER2 = **0.499**, vs SciNCL = 0.505. The most
  diverse candidate ever attempted — yet it still failed.
- Blend probe found exactly one cell with a positive lift, and only
  by +0.0008 (noise band):

  | Mix | OOF round-QWK | Δ vs anchor 0.6074 |
  | --- | ---: | ---: |
  | 60/20/20 anchor | **0.6074** | (baseline) |
  | `ridge -0.05, +openalex 0.05` | 0.6082 | +0.0008 (noise) |
  | ridge -0.10, +openalex 0.10 | 0.6033 | -0.004 |
  | scincl/specter -0.025, +openalex 0.05 | 0.5964 | -0.011 |
  | any weight ≥ 0.10 from any anchor | 0.55-0.60 | clear regression |

Root cause: the label is "ASP/AI-symbolic relevance of an abstract"
(L5), not "paper quality". Citation count and FWCI track impact, not
topic relevance. Heavily-cited proceedings volumes are label 1; niche
label-5 ASP solver papers have near-zero citations. Same trap as the
all-source scholarly dump (L3).

Cross-cutting pattern (4 candidates, same failure mode):

| Candidate | Type | OOF | r vs SPECTER2 | Failure mode |
| --- | --- | ---: | ---: | --- |
| LLM v2 zero-shot | text, generic LLM | 0.375 | 0.558 | low signal |
| SciBERT fine-tune | text, scientific BERT | 0.617 | 0.948 | redundant |
| DeBERTa-v3-large | text, generic large | 0.551 | 0.848 | low signal × medium diversity |
| **OpenAlex anchor** | **non-text** | **0.257** | **0.499** | **low signal even with high diversity** |

L12 codifies this as the "signal floor + diversity floor" rule: a
candidate must clear *both* (OOF ≥ ~0.51 AND r vs SPECTER2 < ~0.95).
OpenAlex passed diversity by a wide margin but failed signal floor by
half. Diversity alone is insufficient.

Updated direction (replacing the row in the bge/openalex table above):

| Idea | Why | Status |
| --- | --- | --- |
| `bge-large-en-v1.5` or `e5-large-v2` | contrastive sentence similarity (closer to SPECTER2's objective than DeBERTa-v3 RTD) | **DONE — public 0.72374** (step 9 + step 10) |
| Re-purpose OpenAlex features as **inputs** to the Ridge anchor | the original 0.63064 OpenAlex Huber meta worked publicly *as a meta-blend*, not as a standalone anchor — feed citation features into Ridge instead of stacking | not yet tried |
| Venue / first-author target encoding (already inside OpenAlex anchor) | re-test by feeding into Ridge alone | not yet tried |
| Qwen2.5-7B LoRA fine-tune | LLM zero-shot lacked signal (L8); fine-tune may differ in reasoning span | bigger bet (3-4h GPU) |


## After 0.72374 — BGE-large 4-anchor stack (NEW BEST, breaks 0.72103)

A 4-anchor weighted blend `bge 0.40 / scincl 0.40 / specter 0.15 /
ridge 0.05` with distribution-constrained threshold tuner pushed Public
LB from `0.72103` to **`0.72374`**, a `+0.00271` improvement. Recipe in
`outputs/step10_bge_stack/safest/`.

This validates BGE-large-en-v1.5 as the **first text encoder anchor
since SciNCL** (May 14) to lift the stack. Both L11/L12 floors cleared:

| Anchor | OOF | Pearson r vs SPECTER2 | Notes |
| --- | ---: | ---: | --- |
| BGE-large-en-v1.5 | 0.6228 | 0.909 | new (step 9) |
| SPECTER2 | 0.6297 | 1.000 | unchanged |
| SciNCL | 0.6117 | 0.943 | unchanged |
| Ridge 5×5 | 0.4353 | 0.811 | unchanged |

Picked by L7/L9 ranking (test L1 first, OOF QWK second):

| Candidate | OOF QWK | Test L1 | Public LB |
| --- | ---: | ---: | ---: |
| anchor (60/20/20, no bge) | 0.6412 | 0.162 | 0.72103 |
| **safest: 40/40/15/05** | **0.6567** | **0.156** | **0.72374** ⭐ |
| single_knob: 30/40/10/20 | 0.6572 | 0.158 | not submitted |
| high_signal: 25/25/25/25 | 0.6593 | 0.172 | not submitted |

The OOF lift was `+0.0155` but only `+0.0027` translated to public —
about a 6:1 OOF-to-public discount. Compare to SciNCL (`0.72103`) which
had OOF lift `−0.0050` but `+0.011` public lift (negative OOF correlation
to public). The two patterns are inconsistent, which is why **L7's
"rank by test L1 first" rule keeps mattering**: the picked candidate had
the lowest test L1 (0.156, *below* the anchor), so the public lift was
modest but positive rather than risking a regression.

### Where to go next

| Idea | Expected gain | Cost |
| --- | --- | --- |
| Submit `single_knob` (30/40/10/20) — test whether higher specter weight helps | ±0.003 | 1 daily slot |
| Submit `high_signal` (25/25/25/25) — test whether higher OOF beats lower test L1 | ±0.005 | 1 daily slot |
| Sweep around the new winner (40/40/15/05): 40/40/10/10, 35/45/15/05, 45/35/15/05 | +0.000 → +0.005 | 1 daily slot |
| `e5-large-v2` fine-tune as a 5th anchor | +0.005 → +0.010 | ~45 min GPU |
| Qwen2.5-7B LoRA fine-tune (L8 fix) | +0.012 → +0.020 (untested) | 3-4h GPU |

Top priority: try `single_knob` next — it has nearly identical OOF to
`safest` (0.6572 vs 0.6567), test_L1 within 0.003 (0.158 vs 0.156), but
keeps ridge at the public-validated 0.20 (per L10). If it scores
≥ 0.72374 we have two confirmation points; if below we know the 0.40
ridge knob doesn't transfer here either.


## After 0.72394 — E5-large-v2 4-anchor stack (NEW BEST, marginal lift; same-family ceiling reached)

A 4-anchor weighted blend `e5 0.50 / scincl 0.30 / specter 0.20 /
ridge 0.00` with distribution-constrained threshold tuner pushed Public
LB from `0.72374` to **`0.72394`**, a **`+0.00020`** improvement (the
smallest positive lift in the project). Recipe in
`outputs/step13_e5_stack/safest_e5/`.

E5 is the strongest single anchor in the project (OOF 0.6417 single-fold
QWK; SPECTER2 0.6297; BGE 0.6228; SciNCL 0.6117) but Pearson r = **0.944
vs BGE** put it in the L7 redundancy zone (r > 0.93 = same signal).
That predicted only marginal stacking gain — confirmed.

**Why we kept E5 and dropped BGE rather than running both:**

| Stack shape | OOF round-QWK | Δ vs anchor |
| --- | ---: | ---: |
| 4-anchor BGE safest (anchor) | 0.6351 | (baseline) |
| 5-anchor (e5 + bge + scincl + specter + ridge), best mix | 0.6364 | +0.0013 |
| **4-anchor E5 swap (e5 replaces bge)**, best mix | **0.6477** | **+0.0126** |

A 5-anchor stack with both contrastive-sentence encoders is dominated
by the 4-anchor stack with the stronger one. Same lesson as L7's
"3 SPECTER2 anchors at r=0.98 = one signal", just at the redundancy
boundary instead of the centre.

**Sweep result (E5 4-anchor, ridge ≤ 0.10, 26 candidates):**

| Pick | weights (e5/scincl/specter/ridge) | OOF QWK | Test L1 | Public LB |
| --- | --- | ---: | ---: | ---: |
| BGE safest (prev best, reference) | bge .40 / scincl .40 / specter .15 / ridge .05 | 0.6567 | 0.156 | 0.72374 |
| **safest_e5** ⭐ | 0.50 / 0.30 / 0.20 / **0.00** | **0.6593** | **0.142** | **0.72394** |
| high_oof_e5 (not submitted) | 0.50 / 0.25 / 0.20 / 0.05 | 0.6615 | 0.149 | not submitted |
| swap_bge_e5 (not submitted) | 0.40 / 0.40 / 0.15 / 0.05 | 0.6577 | 0.162 | not submitted |

Picked by L7/L9 (test L1 first): `safest_e5` had test_L1 0.142, the
lowest in any sweep we've ever run. Ridge=0.00 was unusual — every
prior public-best had ridge > 0 — but the L1 floor justified trusting
the rank.

**OOF→public ratio is widening:**

| Submission | OOF lift | Public lift | Ratio |
| --- | ---: | ---: | ---: |
| BGE 4-anchor safest (vs 0.72103 anchor) | +0.0155 | +0.0027 | 6× |
| **E5 4-anchor safest_e5 (vs BGE safest)** | +0.0026 | +0.0002 | **13×** |

We are at the test ceiling for the contrastive-sentence-similarity
family. Each additional +0.001 OOF inside this family buys roughly
+0.0001 public LB. **Adding more BERT-large encoders trained with
contrastive sentence similarity will not lift further.**

### Where to go next

| Idea | Why | Expected lift | Cost |
| --- | --- | --- | --- |
| **Qwen2.5-7B / Llama-3.1-8B LoRA fine-tune** | generative regression — fundamentally different geometry from CLS-mean encoders; expected r vs E5 in 0.6-0.8 band | +0.005 → +0.020 | 3-4h A100 |
| Cross-encoder fine-tune (e.g., `cross-encoder/ms-marco-MiniLM`) | ranks (paper, ASP-relevance) jointly instead of embedding the paper | +0.003 → +0.010 | ~30 min |
| Pseudo-labeling round | re-train E5 on confident test predictions; risk per L4 (data drift) | ±0.005 | ~1.5h |
| ~~bge-m3 / gte-large / nomic-embed / mxbai-embed-large~~ | ~~more contrastive-sentence encoders~~ | ~~~+0.000~~ | ~~don't bother (L15)~~ |

Top priority is **LLM LoRA**. Three text encoders in the same family
(SPECTER2 + SciNCL + E5 + BGE) confirm that this signal source has
reached its ceiling on this dataset. Only a structurally different
modeling approach (generative regression, cross-encoder, pseudo-labels)
can push further.
