# EDA v2 — ASP-relevance Hypothesis

## 1. Keyword groups vs Label (train)

| keyword_group    |   n_with |   n_without |   label_mean_with |   label_mean_without |   label_mean_delta |   share_label_5_with |   share_label_5_without |   share_label_1_with |   share_label_1_without |   label_overall_mean |
|:-----------------|---------:|------------:|------------------:|---------------------:|-------------------:|---------------------:|------------------------:|---------------------:|------------------------:|---------------------:|
| asp_core         |      227 |        2267 |             3.916 |                2.287 |              1.63  |                0.396 |                   0.08  |                0.048 |                   0.393 |                2.435 |
| ai_symbolic      |      199 |        2295 |             3.563 |                2.337 |              1.226 |                0.372 |                   0.086 |                0.136 |                   0.382 |                2.435 |
| asp_ecosystem    |       36 |        2458 |             3.639 |                2.417 |              1.221 |                0.361 |                   0.105 |                0.083 |                   0.366 |                2.435 |
| logic_neighbour  |      234 |        2260 |             2.726 |                2.405 |              0.322 |                0.081 |                   0.112 |                0.231 |                   0.376 |                2.435 |
| low_signal_title |       87 |        2407 |             1.391 |                2.473 |             -1.082 |                0.011 |                   0.113 |                0.793 |                   0.346 |                2.435 |

Interpretation:

- **asp_core** / **asp_ecosystem** with positive `label_mean_delta` and high `share_label_5_with` confirms ASP-related papers are pushed toward higher labels.
- **low_signal_title** with negative delta confirms proceedings / short paper / abstract titles cluster at label 1.
- **ai_symbolic** delta tells whether neural / ML / explainability cues drive label 4-5 across venues, which is what cav/lics label-5 examples suggested.
- **logic_neighbour** delta tells whether non-ASP logic topics (Argumentation, Description Logic, ...) drift toward lower labels at iclp/lpnmr; this matters because the model needs to learn 'logic but not ASP' is not the same as 'ASP'.

## 2. Keyword × venue (train)

| keyword_group    | venue   |   n_with |   n_without |   label_mean_with |   label_mean_without |   label_mean_delta |   share_label_5_with |   share_label_1_with |
|:-----------------|:--------|---------:|------------:|------------------:|---------------------:|-------------------:|---------------------:|---------------------:|
| ai_symbolic      | cav     |       68 |         589 |             3.5   |                1.961 |              1.539 |                0.294 |                0.132 |
| ai_symbolic      | lpnmr   |        9 |         107 |             4.444 |                3.028 |              1.416 |                0.556 |                0     |
| ai_symbolic      | kr      |       53 |         608 |             3.623 |                2.651 |              0.971 |                0.415 |                0.151 |
| ai_symbolic      | lics    |       13 |         591 |             2.846 |                1.917 |              0.929 |                0.308 |                0.231 |
| ai_symbolic      | iclp    |       56 |         400 |             3.607 |                2.85  |              0.757 |                0.411 |                0.125 |
| asp_core         | iclp    |      141 |         315 |             3.901 |                2.514 |              1.386 |                0.418 |                0.05  |
| asp_core         | lpnmr   |       47 |          69 |             3.957 |                2.58  |              1.378 |                0.277 |                0.043 |
| asp_core         | kr      |       39 |         622 |             3.923 |                2.654 |              1.269 |                0.462 |                0.051 |
| asp_ecosystem    | lpnmr   |       10 |         106 |             4.1   |                3.047 |              1.053 |                0.4   |                0     |
| asp_ecosystem    | iclp    |       16 |         440 |             3.688 |                2.916 |              0.772 |                0.438 |                0.125 |
| asp_ecosystem    | kr      |       10 |         651 |             3.1   |                2.724 |              0.376 |                0.2   |                0.1   |
| logic_neighbour  | kr      |      130 |         531 |             2.785 |                2.716 |              0.069 |                0.062 |                0.208 |
| logic_neighbour  | lics    |        2 |         602 |             2     |                1.937 |              0.063 |                0     |                0     |
| logic_neighbour  | iclp    |       73 |         383 |             2.74  |                2.982 |             -0.242 |                0.11  |                0.219 |
| logic_neighbour  | lpnmr   |       26 |          90 |             2.615 |                3.289 |             -0.674 |                0.115 |                0.346 |
| logic_neighbour  | cav     |        3 |         654 |             1.333 |                2.124 |             -0.791 |                0     |                0.667 |
| low_signal_title | lics    |        8 |         596 |             1.625 |                1.941 |             -0.316 |                0     |                0.625 |
| low_signal_title | cav     |       30 |         627 |             1.033 |                2.172 |             -1.139 |                0     |                0.967 |
| low_signal_title | kr      |       18 |         643 |             1.611 |                2.76  |             -1.149 |                0     |                0.667 |
| low_signal_title | iclp    |       25 |         431 |             1.68  |                3.016 |             -1.336 |                0.04  |                0.68  |
| low_signal_title | lpnmr   |        6 |         110 |             1     |                3.255 |             -2.255 |                0     |                1     |

Reading: even within the same venue, the ASP keyword should still increase the mean label. If it does, ASP signal is venue-independent — exactly what we need a semantic embedding to capture, since TF-IDF n-grams already exploit some of this but cannot generalize across paraphrases like 'ASP' vs 'answer set programming' vs 'stable model'.

## 3. Missing authors vs Label

| group           |    n |   label_mean |   share_label_5 |   share_label_1 |
|:----------------|-----:|-------------:|----------------:|----------------:|
| has_authors     | 2302 |        2.448 |           0.109 |           0.353 |
| missing_authors |  192 |        2.281 |           0.109 |           0.474 |

If `missing_authors` clusters at label 1, that matches the proceedings/abstract pattern and explains why the current Ridge anchor uses `authors_clean` as a useful but not dominant feature.

## 4. Cross-split duplicate titles

No exact normalized title overlap between train and public/private. Test rows = 596.

## 5. First-author leakage

| split   |   test_rows |   with_first_author_train_match |   share_with_first_author_train_match |   single_train_paper_anchor_rows |   share_single_anchor |   train_count_mean_when_matched |   train_label_std_mean_when_matched |
|:--------|------------:|--------------------------------:|--------------------------------------:|---------------------------------:|----------------------:|--------------------------------:|------------------------------------:|
| public  |         298 |                             122 |                                 0.409 |                               39 |                 0.131 |                           3.77  |                               0.758 |
| private |         298 |                             132 |                                 0.443 |                               49 |                 0.164 |                           3.288 |                               0.63  |

If a large share of test rows share a first-author surname with train, the existing TF-IDF model already exploits that. SPECTER2 will not lose that signal because we will keep `authors_clean` as a parallel TF-IDF feature in the stack.

## 6. Keyword coverage on the test splits

| split   | keyword_group    |   share_with |   n_with |   n_total |
|:--------|:-----------------|-------------:|---------:|----------:|
| public  | asp_core         |        0.034 |       10 |       298 |
| public  | asp_ecosystem    |        0.013 |        4 |       298 |
| public  | logic_neighbour  |        0.064 |       19 |       298 |
| public  | ai_symbolic      |        0.087 |       26 |       298 |
| public  | low_signal_title |        0.044 |       13 |       298 |
| private | asp_core         |        0.05  |       15 |       298 |
| private | asp_ecosystem    |        0.007 |        2 |       298 |
| private | logic_neighbour  |        0.05  |       15 |       298 |
| private | ai_symbolic      |        0.06  |       18 |       298 |
| private | low_signal_title |        0.05  |       15 |       298 |

This tells us how often each ASP / AI-symbolic cue appears on public vs private. If coverage is balanced, a semantic model that captures these cues should transfer cleanly between public and private leaderboard halves.

## 7. Sanity probe — rule-based proxy QWK

- Proxy QWK on train: **0.2589** (no model fit, only the keyword rules above).
- Proxy label distribution: `{1: 77, 2: 203, 3: 1830, 4: 156, 5: 228}`
- True label distribution:  `{1: 903, 2: 514, 3: 438, 4: 367, 5: 272}`

Reading:

- A non-trivial proxy QWK from 5 hand-written keyword groups is direct evidence that Label 1-5 is **content / topic relevance**, not a quality score. TF-IDF already approximates this with thousands of features, but it cannot generalise the way a sentence transformer can. This is exactly what we exploit in step 3.
- The proxy distribution is intentionally peaked at label 3 because most papers do not match any of the curated keyword groups. Once we move to SPECTER2 embeddings, the signal will spread across all 5 labels because the embedding captures the rest of the paper's topic, not just hand-picked keywords.

## 8. Decision before step 2

If the keyword-group table shows that ASP-core papers have a clearly higher mean label than non-ASP papers within the same venue, we proceed with the SPECTER2 + abstract crawl plan. The remaining steps are:

1. Crawl `abstract` and `tldr` from Semantic Scholar (already partially cached) and OpenAlex.
2. Encode `title (+ abstract)` with allenai/specter2_base or sentence-transformers/all-MiniLM-L6-v2 as a CPU fallback.
3. Train a Ridge ordinal head on the embeddings, repeat the 5x5 CV pattern, threshold-tune on OOF.
4. Add an LLM zero/few-shot 1-5 score as a meta feature.
5. Stack with the existing 0.62820 Ridge anchor and 0.63064 OpenAlex anchor.