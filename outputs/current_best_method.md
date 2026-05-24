# Current Best Method

Updated: 2026-05-24

## Best Public Score

- Public score: **`0.73054`**
- Submission: `outputs/submissions/next_step25_bge_m3_best_safe_submission.csv`
- Archive: `outputs/submissions/next_step25_bge_m3_best_safe_submission.csv`
- Method artifacts: `outputs/bge_m3_frozen_anchor/best_safe/`
- Candidate sweep: `outputs/bge_m3_frozen_anchor/`
- Previous anchor: Step 18b targeted calibration scored `0.72808`
- Jump over previous anchor: **`+0.00246`**

## Method Summary

Step 25 BGE-M3 frozen-anchor diagnostic blend on top of the Step 18b targeted feature calibration.

Base Step18b score:

```text
step18b = 0.92 * safest_e5 + 0.08 * Ridge(alpha=10) pattern calibrator
```

Step25 diagnostic blend:

```text
candidate = step18b_bge_m3_ridge_a30p0_w0p02
final_score = 0.98 * step18b + 0.02 * BGE-M3 Ridge(alpha=30) diagnostic correction
```

Then the same distribution-constrained threshold tuner maps the blended score to labels 1-5, keeping the prediction distribution inside the proven Step18b safe band.

## Why this is the current best

1. **Step18b remains the backbone.** The BGE-M3 correction weight is only `0.02`, so the public-proven targeted calibrator dominates.
2. **BGE-M3 adds a small out-of-family encoder diagnostic.** It is not a broad replacement stack; it only moves a few near-threshold rows.
3. **The winning candidate stayed inside the L17 safe distribution band.** `test_L1=0.145552`, below the hard cap `0.147`.
4. **The local lift was small but transferred.** OOF improved from `0.660176` to `0.661261`; public improved from `0.72808` to `0.73054`.

## OOF + test metrics

```text
OOF QWK (constrained tuner) = 0.661261
OOF lift vs Step18b        = +0.001084
Test combined L1 vs train  = 0.145552
Changed test rows vs base  = 9
Public LB                  = 0.73054
```

Previous Step18b baseline:

```text
OOF QWK (constrained tuner) = 0.660176
Test combined L1 vs train  = 0.145552
Public LB                  = 0.72808
```

## Selected blend

| Component | Weight | Role |
| --- | ---: | --- |
| Step 18b targeted calibration | 0.98 | public-proven semantic + OOF-pattern backbone |
| Step 25 BGE-M3 diagnostic correction | 0.02 | low-weight frozen encoder signal for near-threshold rows |

Base-stack anchor weights:

| Anchor | Base weight | Role |
| --- | ---: | --- |
| E5-large | 0.50 | strongest safe encoder anchor |
| SciNCL | 0.30 | scientific neighborhood contrastive signal |
| SPECTER2 | 0.20 | scientific triplet/paper-similarity signal |
| Ridge TF-IDF | 0.00 | retained as reference, not used in base stack |

## Reproduce

```bash
python src/step18_oof_error_audit.py
python src/step18b_targeted_feature_calibration.py
python src/step25_bge_m3_stack.py
```

Expected primary output:

```text
outputs/submissions/next_step25_bge_m3_best_safe_submission.csv
outputs/bge_m3_frozen_anchor/best_safe/
```

## Anchors kept for future stacking

| Folder | Public LB | OOF QWK | Role |
| --- | ---: | ---: | --- |
| `outputs/0.62820/` | 0.62820 | 0.5921 | lexical Ridge TF-IDF baseline |
| `outputs/0.69972/` | 0.69972 | 0.6373 | SPECTER2 fine-tune |
| `outputs/scincl_finetune/` | not submitted | 0.6269 | SciNCL fine-tune |
| `outputs/0.72103/` | 0.72103 | 0.6412 | earlier SciNCL/SPECTER2/Ridge stack |
| `outputs/0.72394/` | 0.72394 | 0.6593 | previous E5 4-anchor stack |
| `outputs/0.72808/` | 0.72808 | 0.6602 | former best Step 18b targeted calibration |
| `outputs/bge_m3_frozen_anchor/` | **0.73054** | **0.6613** | **current best Step 25 BGE-M3 frozen diagnostic anchor** |
| `outputs/qwen_lora_finetune_fixed_outputs/` | not submitted alone | 0.6205 | fixed Qwen LoRA anchor, useful as negative/shift reference |
| `outputs/scholarly_graph_anchor/` | not submitted | 0.3633 | metadata/graph anchor, diverse but too weak |

## Current decision rule

Use the Step 25 BGE-M3 diagnostic submission as the default public-best baseline. New candidates should not replace it unless they either:

- beat `0.73054` publicly, or
- improve local OOF while staying inside the safe distribution band around `test_L1 ≈ 0.1456`, changing only a small number of rows unless there is a new validated rule family.

Recent failed or lower-priority promotion checks:

| Experiment | Best local sign | Why not promoted |
| --- | --- | --- |
| Qwen fixed 5-anchor Step 15b | Qwen-bearing blends reached OOF about `0.6653` | no Qwen-bearing candidate stayed inside L17 safe cap `test_L1 <= 0.147` |
| Qwen14B Step 17c | OOF up to `0.6760` | public regressed to `0.714xx`; old notebook had PEFT adapter-chaining warnings |
| Clean Qwen14B Step 22 | OOF up to `0.6687` with safe `test_L1=0.1456` | public regressed to `0.72403` main / `0.72106` hedge; fresh-base fix removed leakage but the generative-LM anchor still overfits public changed-row directions |
| Qwen14B disagreement gate Step 23 | 12-row fixed-threshold gate reached OOF `0.6631`, 12/12 OOF improvements, `test_L1=0.1456` | public tied Step18b at `0.72808` but did not beat it; keep as private hedge, not new main |
| Scholarly graph + E5 Step 16 | graph/E5 correlation only `r ≈ 0.507` | standalone graph OOF only `0.3633`; safe graph blend lowered E5 OOF |
| Step 18b heavier tree blends | OOF up to `0.6616` | test_L1 drifted outside safe cap |
