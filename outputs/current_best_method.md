# Current Best Method

Updated: 2026-05-20

## Best Public Score

- Public score: **`0.72394`**
- Submission: `outputs/submissions/next_e5_4anchor_safest_e5_submission.csv`
- Archive: `outputs/0.72394/`
- Method artifacts: `outputs/step13_e5_stack/safest_e5/`
- Candidate sweep: `outputs/step13_e5_stack/candidates.csv`
- Previous anchor: BGE safest stack scored `0.72374`
- Jump over previous anchor: **`+0.00020`**

## Method Summary

Step 13 E5 4-anchor weighted stack:

```text
blended_score = 0.50 * e5 + 0.30 * scincl + 0.20 * specter2 + 0.00 * ridge
```

Then a distribution-constrained threshold tuner maps the blended score to labels 1-5, with the same train-distribution L1 penalty used in the safer post-L7/L9 sweeps.

## Why this is the current best

1. **E5 replaced BGE as the strongest safe encoder anchor.** E5-large provided the best public-validated 4-anchor stack while keeping test distribution closest to train.
2. **The safest E5 stack won public transfer, not the highest local novelty.** It had the best known safe `test_L1` band among major submitted stacks.
3. **Same-family 5-anchor expansion was rejected.** E5+BGE redundancy was too high (`r ≈ 0.944`), so adding BGE back as a fifth encoder did not justify the extra drift risk.
4. **Later Qwen and metadata/graph experiments did not clear the safe-transfer bar.** Qwen improved some local OOF blends but shifted test distribution; scholarly graph metadata had useful diversity but too weak standalone label signal.

## OOF + test metrics

```text
OOF QWK (constrained tuner) = 0.6593
OOF round-QWK              = 0.6475
Test combined L1 vs train  = 0.1422
Public LB                  = 0.72394
```

Selected weights:

| Anchor | Weight | Role |
| --- | ---: | --- |
| E5-large | 0.50 | strongest safe encoder anchor |
| SciNCL | 0.30 | scientific neighborhood contrastive signal |
| SPECTER2 | 0.20 | scientific triplet/paper-similarity signal |
| Ridge TF-IDF | 0.00 | retained as reference, not used in best stack |

## Reproduce

```bash
python src/step13_e5_anchor_stack.py
```

Expected primary output:

```text
outputs/submissions/next_e5_4anchor_safest_e5_submission.csv
outputs/step13_e5_stack/safest_e5/
outputs/step13_e5_stack/candidates.csv
```

## Anchors kept for future stacking

| Folder | Public LB | OOF QWK | Role |
| --- | ---: | ---: | --- |
| `outputs/0.62820/` | 0.62820 | 0.5921 | lexical Ridge TF-IDF baseline |
| `outputs/0.69972/` | 0.69972 | 0.6373 | SPECTER2 fine-tune |
| `outputs/scincl_finetune/` | not submitted | 0.6269 | SciNCL fine-tune |
| `outputs/0.72103/` | 0.72103 | 0.6412 | earlier SciNCL/SPECTER2/Ridge stack |
| `outputs/step13_e5_stack/safest_e5/` | **0.72394** | **0.6593** | **current best E5 4-anchor stack** |
| `outputs/qwen_lora_finetune_fixed_outputs/` | not submitted alone | 0.6205 | fixed Qwen LoRA anchor, useful as negative/shift reference |
| `outputs/scholarly_graph_anchor/` | not submitted | 0.3633 | metadata/graph anchor, diverse but too weak |

## Current decision rule

Use the E5 4-anchor safest stack as the default submission baseline. New candidates should not replace it unless they either:

- beat `0.72394` publicly, or
- improve local OOF while staying inside the safe distribution band around `test_L1 ≈ 0.1422`.

Recent failed promotion checks:

| Experiment | Best local sign | Why not promoted |
| --- | --- | --- |
| Qwen fixed 5-anchor Step 15b | Qwen-bearing blends reached OOF about `0.6653` | no Qwen-bearing candidate stayed inside L17 safe cap `test_L1 <= 0.147` |
| Scholarly graph + E5 Step 16 | graph/E5 correlation only `r ≈ 0.507` | standalone graph OOF only `0.3633`; safe graph blend lowered E5 OOF |
