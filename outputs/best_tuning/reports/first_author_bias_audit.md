# First-Author Bias Audit

This audit checks whether `first_author_surname` can over-bias predictions.

- Test first-author groups: 473
- Seen in train: 196
- Unseen in train: 277
- Train count = 1 and test count >= 2: 9
- Train count = 1 with label 5 and test count >= 2: 0
- Pure-label train author with train count >= 2 and test count >= 2: 1
- Pure-label train authors where all test predictions match the only train label: 37

## Interpretation

- The model can learn author tokens, so rare-author leakage-like bias is possible.
- The audit separates risky author groups from normal title-driven predictions.
- If public score drops for a higher-CV author-heavy model, prefer compact/regularized author features.
