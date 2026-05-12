# OpenAlex Title Meta Candidates

This run expands DOI-only OpenAlex enrichment with title-search matches for rows without DOI.

## Coverage

| source_split   |   doi |   none |   title |
|:---------------|------:|-------:|--------:|
| private_test   |   182 |     23 |      93 |
| public_test    |   177 |     21 |     100 |
| train          |  1455 |    297 |     742 |

## Top Candidates

| name                            | base_model   |   meta_weight |   oof_qwk |      mae | thresholds                                                  | train_distribution                        | test_distribution                      | submission                                                                                                                                      |
|:--------------------------------|:-------------|--------------:|----------:|---------:|:------------------------------------------------------------|:------------------------------------------|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| openalex_title_huber_meta_w0.50 | huber_meta   |          0.5  |  0.603484 | 0.856055 | 2.084392503764,2.390949017632,2.877210994435,3.340176485211 | {1: 917, 2: 497, 3: 477, 4: 293, 5: 310}  | {1: 252, 2: 124, 3: 119, 4: 45, 5: 56} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_huber_meta_w0.50_submission.csv |
| openalex_title_huber_meta_w0.15 | huber_meta   |          0.15 |  0.602993 | 0.859262 | 2.074064861974,2.348823904917,2.864007410052,3.172218396398 | {1: 805, 2: 511, 3: 603, 4: 229, 5: 346}  | {1: 222, 2: 129, 3: 150, 4: 36, 5: 59} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_huber_meta_w0.15_submission.csv |
| openalex_title_ridge_meta_w0.50 | ridge_meta   |          0.5  |  0.60264  | 0.852045 | 2.103617018537,2.574534700993,2.963128392192,3.261999691487 | {1: 899, 2: 689, 3: 367, 4: 191, 5: 348}  | {1: 248, 2: 183, 3: 72, 4: 31, 5: 62}  | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_ridge_meta_w0.50_submission.csv |
| openalex_title_ridge_meta_w0.25 | ridge_meta   |          0.25 |  0.602568 | 0.851243 | 2.079047666617,2.351716527238,2.889473594656,3.289086835122 | {1: 809, 2: 509, 3: 613, 4: 265, 5: 298}  | {1: 222, 2: 129, 3: 154, 4: 38, 5: 53} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_ridge_meta_w0.25_submission.csv |
| openalex_title_huber_meta_w0.35 | huber_meta   |          0.35 |  0.602317 | 0.853248 | 2.121620683213,2.521656624854,2.843669547461,3.281195149706 | {1: 937, 2: 615, 3: 329, 4: 296, 5: 317}  | {1: 258, 2: 165, 3: 71, 4: 48, 5: 54}  | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_huber_meta_w0.35_submission.csv |
| openalex_title_huber_meta_w1.00 | huber_meta   |          1    |  0.6021   | 0.842823 | 1.998955194503,2.550815837576,2.978852536344,3.654899184004 | {1: 936, 2: 637, 3: 340, 4: 328, 5: 253}  | {1: 246, 2: 176, 3: 71, 4: 52, 5: 51}  | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_huber_meta_w1.00_submission.csv |
| openalex_title_ridge_meta_w0.35 | ridge_meta   |          0.35 |  0.601494 | 0.855654 | 2.148259676417,2.480533169444,2.910897996411,3.254187644517 | {1: 950, 2: 535, 3: 447, 4: 234, 5: 328}  | {1: 262, 2: 137, 3: 103, 4: 39, 5: 55} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_ridge_meta_w0.35_submission.csv |
| openalex_title_huber_meta_w0.75 | huber_meta   |          0.75 |  0.601431 | 0.84603  | 2.091681588783,2.480607179424,2.941662009145,3.575701158716 | {1: 1001, 2: 510, 3: 400, 4: 331, 5: 252} | {1: 273, 2: 129, 3: 94, 4: 50, 5: 50}  | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_huber_meta_w0.75_submission.csv |
| openalex_title_huber_meta_w0.25 | huber_meta   |          0.25 |  0.601019 | 0.871692 | 2.125885187536,2.366586849505,2.885136141729,3.144062510534 | {1: 927, 2: 434, 3: 559, 4: 189, 5: 385}  | {1: 249, 2: 117, 3: 138, 4: 29, 5: 63} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_huber_meta_w0.25_submission.csv |
| openalex_title_ridge_meta_w0.75 | ridge_meta   |          0.75 |  0.600791 | 0.850441 | 2.131627055235,2.447151703813,3.022532659866,3.484058724390 | {1: 987, 2: 450, 3: 529, 4: 254, 5: 274}  | {1: 268, 2: 114, 3: 123, 4: 39, 5: 52} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_ridge_meta_w0.75_submission.csv |
| openalex_title_ridge_meta_w0.15 | ridge_meta   |          0.15 |  0.599705 | 0.849238 | 2.068668902380,2.467556212897,2.973848674505,3.181312272672 | {1: 768, 2: 707, 3: 528, 4: 154, 5: 337}  | {1: 217, 2: 179, 3: 126, 4: 16, 5: 58} | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_ridge_meta_w0.15_submission.csv |
| openalex_title_ridge_meta_w1.00 | ridge_meta   |          1    |  0.594899 | 0.845229 | 2.127829708209,2.341825977829,3.143303043195,3.790441092972 | {1: 1009, 2: 294, 3: 708, 4: 289, 5: 194} | {1: 272, 2: 78, 3: 163, 4: 44, 5: 39}  | C:\Users\LG\Desktop\Study Material\DataMining\Assignment\outputs\openalex_title_meta\submissions\openalex_title_ridge_meta_w1.00_submission.csv |

## Notes

- `meta_weight=1.0` uses only the OpenAlex meta-model score.
- Smaller `meta_weight` values are light blends with the Ridge 5x5 score anchor used by the `0.62820` submission.
- Citation velocity and venue/year percentile features are included to reduce raw citation-count bias.
