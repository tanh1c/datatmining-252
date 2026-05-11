# EDA Report

## Dataset Shape

- Train rows: 2494
- Public test rows: 298
- Private test rows: 298
- Label range: [1, 2, 3, 4, 5]

## Label Distribution

|   Label |   count |   percent |
|--------:|--------:|----------:|
|       1 |     903 |     36.21 |
|       2 |     514 |     20.61 |
|       3 |     438 |     17.56 |
|       4 |     367 |     14.72 |
|       5 |     272 |     10.91 |

## Missingness

| column   |   missing |   blank_or_missing |   blank_or_missing_percent |
|:---------|----------:|-------------------:|---------------------------:|
| title    |         0 |                  0 |                        0   |
| venue    |         0 |                  0 |                        0   |
| year     |         0 |                  0 |                        0   |
| authors  |       192 |                192 |                        7.7 |
| doi      |         0 |                  0 |                        0   |
| Label    |         0 |                  0 |                        0   |

## Strong Data Signals

- Venues are limited: 5 unique train venues.
- Most common venue: kr (661 rows).
- Duplicate normalized titles in train: 6 groups.
- Duplicate title groups with conflicting labels: 2.

## Train/Test Coverage

| split   |   rows |   title_exact_matches_train |   venue_unseen_count |   year_unseen_count |   first_author_matches_train |
|:--------|-------:|----------------------------:|---------------------:|--------------------:|-----------------------------:|
| public  |    298 |                           0 |                    0 |                   0 |                          150 |
| private |    298 |                           0 |                    0 |                   0 |                          158 |

## Feature Ablation

| variant                           | vectorizer   |   cv_accuracy |   cv_qwk |
|:----------------------------------|:-------------|--------------:|---------:|
| title_first_author                | word_char    |      0.413392 | 0.526064 |
| title_venue_year                  | word_char    |      0.410986 | 0.524941 |
| title_x2                          | word_char    |      0.409383 | 0.522345 |
| title_only                        | word_char    |      0.410585 | 0.520687 |
| title_venue_year_authors          | word_char    |      0.407378 | 0.517585 |
| title_venue_year_authors_doi_kind | word_char    |      0.402967 | 0.512902 |
| title_venue_year_authors          | word         |      0.392542 | 0.501585 |
| title_x2                          | word         |      0.395349 | 0.501538 |
| title_only                        | word         |      0.393745 | 0.500306 |
| title_venue_year                  | word         |      0.392542 | 0.499875 |
| title_first_author                | word         |      0.392943 | 0.498836 |
| title_venue_year_authors_doi_kind | word         |      0.389334 | 0.49668  |

## Actionable Takeaways

1. Logistic TF-IDF remains the right search area; it beats the RandomForest baseline in CV and public submissions.
2. Title text is the core signal. Metadata-only is weaker, so metadata should be added as small tokens rather than dominating the text.
3. Venue/year tokens help but can overfit; keep them prefixed and test against public/private drift.
4. Author information is useful mainly as a compact cue. Try first-author surname rather than the full author list.
5. `doi` should not be used as raw text. If used, keep only coarse `doi_kind` tokens because DOI strings are noisy identifiers.
6. Since public leaderboard is only half of test data, favor simple Logistic variants with stable CV over aggressive blends.
