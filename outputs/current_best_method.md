# Current Best Method

Updated: 2026-05-24

## Best Public Score

- Public score: **`0.73213`**
- Submission: `outputs/submissions/next_step36_risky_small_diff_ridge_a3_w0p0275_l4_submission.csv`
- Archive: `outputs/submissions/next_step36_risky_small_diff_ridge_a3_w0p0275_l4_submission.csv`
- Method artifacts: `outputs/step36_bge_m3_fine_sweep/`
- Previous anchor: Step 25 BGE-M3 safe frozen diagnostic scored `0.73054`
- Jump over previous anchor: **`+0.00159`**

## Method Summary

Step 36 BGE-M3 fine sweep on top of the Step 18b targeted feature calibration superseded Step25 by moving from the original safe `ridge_a30, w=0.02` correction to a slightly stronger but still small-diff `ridge_a3, w=0.0275` correction.

Base Step18b score:

```text
step18b = 0.92 * safest_e5 + 0.08 * Ridge(alpha=10) pattern calibrator
```

Step36 winning diagnostic blend:

```text
candidate = step18b_bge_m3_ridge_a3p0_w0p0275_l4p0
final_score = 0.9725 * step18b + 0.0275 * BGE-M3 Ridge(alpha=3) diagnostic correction
threshold_lambda = 4.0
```

The candidate was originally labeled risky because its `test_L1=0.148908` exceeded the strict Step25 safe cap `0.147`, but it changed only 5 total test rows versus Step25 and transferred best on public.

## Why this is the current best

1. **Step18b remains the backbone.** The BGE-M3 correction weight is only `0.0275`, so the public-proven targeted calibrator still dominates.
2. **BGE-M3 remains a small diagnostic, not a broad replacement stack.** The winning probe changed only 5 test rows versus Step25.
3. **Small diff beat higher OOF.** The highest-OOF Step36 probes did not win public; the closest-to-Step25 probe did.
4. **Strict safe gates were too conservative here.** `test_L1=0.148908` was slightly above the old cap, but the small changed-row count made it worth an explicitly risky public probe.

## OOF + test metrics

```text
OOF QWK (constrained tuner) = 0.662098
OOF lift vs Step25        = +0.000837
Test combined L1 vs train = 0.148908
Diff vs Step25 total      = 5
Public LB                 = 0.73213
```

Previous Step25 baseline:

```text
OOF QWK (constrained tuner) = 0.661261
Test combined L1 vs train  = 0.145552
Public LB                  = 0.73054
```

Step36 public probes:

| Probe | OOF QWK | test_L1 | Diff vs Step25 | Public |
| --- | ---: | ---: | ---: | ---: |
| `ridge_a3p0_w0p0300_l4p0` | 0.663105 | 0.145552 | 9 | 0.72602 |
| `ridge_a3p0_w0p0275_l4p0` | 0.662098 | 0.148908 | 5 | **0.73213** |
| `ridge_a3p0_w0p0375_l4p0` | 0.663312 | 0.148908 | 11 | 0.73201 |

## Selected blend

| Component | Weight | Role |
| --- | ---: | --- |
| Step 18b targeted calibration | 0.9725 | public-proven semantic + OOF-pattern backbone |
| Step 36 BGE-M3 Ridge(alpha=3) diagnostic correction | 0.0275 | low-weight frozen encoder signal for near-threshold rows |

Base-stack anchor weights inside Step18b:

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
python src/step36_bge_m3_fine_sweep.py
python src/step36_export_risky_probes.py
```

Expected primary output:

```text
outputs/submissions/next_step36_risky_small_diff_ridge_a3_w0p0275_l4_submission.csv
outputs/step36_bge_m3_fine_sweep/risky_probe_summary.json
```

## Anchors kept for future stacking

| Folder | Public LB | OOF QWK | Role |
| --- | ---: | ---: | --- |
| `outputs/0.62820/` | 0.62820 | 0.5921 | lexical Ridge TF-IDF baseline |
| `outputs/0.69972/` | 0.69972 | 0.6389 | SPECTER2 fine-tune |
| `outputs/scincl_finetune/` | not submitted | 0.6269 | SciNCL fine-tune |
| `outputs/0.72103/` | 0.72103 | 0.6412 | earlier SciNCL/SPECTER2/Ridge stack |
| `outputs/0.72394/` | 0.72394 | 0.6593 | previous E5 4-anchor stack |
| `outputs/0.72808/` | 0.72808 | 0.6602 | Step18b targeted calibration |
| `outputs/bge_m3_frozen_anchor/` | 0.73054 | 0.6613 | Step25 BGE-M3 frozen diagnostic anchor |
| `outputs/step36_bge_m3_fine_sweep/` | **0.73213** | **0.6621** | **current best Step36 BGE-M3 fine-sweep diagnostic anchor** |
| `outputs/qwen_lora_finetune_fixed_outputs/` | not submitted alone | 0.6205 | fixed Qwen LoRA anchor, useful as negative/shift reference |
| `outputs/scholarly_graph_anchor/` | not submitted | 0.3633 | metadata/graph anchor, diverse but too weak |

## Current decision rule

Use the Step36 small-diff BGE-M3 diagnostic submission as the default public-best baseline. New candidates should not replace it unless they either:

- beat `0.73213` publicly, or
- improve local OOF while staying close to the Step36/Step25 distribution band and changing only a very small number of rows unless there is a new validated rule family.

Recent failed or lower-priority promotion checks:

| Experiment | Best local sign | Why not promoted |
| --- | --- | --- |
| Step35D safe demotion gate | one-row OOF-safe demotion over Step25 | public regressed to `0.72820`; do not demote CAV/formal synthesis row 2022 to label 1 |
| Step36 high-OOF `w=0.0300` probe | OOF `0.663105`, `test_L1=0.145552` | public regressed to `0.72602`; higher OOF alone is not enough |
| Step36 high-OOF `w=0.0375` probe | highest OOF `0.663312` | public `0.73201`, slightly below small-diff winner despite stronger OOF |
| Step37 micro-sweep `w=0.0280,l6` | only 1 row diff vs Step36 and OOF `0.662340` | public regressed to `0.72879`; one-row nearby retune still harmful |
| Step37 high-OOF micro-sweep `w=0.0290,l6` | OOF `0.663513`, highest in local BGE-M3 micro-sweep | public regressed to `0.72916`; stop blind weight/threshold micro-sweeps around Step36 |
| Qwen fixed 5-anchor Step 15b | Qwen-bearing blends reached OOF about `0.6653` | no Qwen-bearing candidate stayed inside L17 safe cap `test_L1 <= 0.147` |
| Qwen14B Step 17c | OOF up to `0.6760` | public regressed to `0.714xx`; old notebook had PEFT adapter-chaining warnings |
| Clean Qwen14B Step 22 | OOF up to `0.6687` with safe `test_L1=0.1456` | public regressed to `0.72403` main / `0.72106` hedge; fresh-base fix removed leakage but the generative-LM anchor still overfits public changed-row directions |
| Qwen14B disagreement gate Step 23 | 12-row fixed-threshold gate reached OOF `0.6631`, 12/12 OOF improvements, `test_L1=0.1456` | public tied Step18b at `0.72808` but did not beat it; keep as private hedge, not new main |
| Scholarly graph + E5 Step 16 | graph/E5 correlation only `r ≈ 0.507` | standalone graph OOF only `0.3633`; safe graph blend lowered E5 OOF |
| Step 18b heavier tree blends | OOF up to `0.6616` | test_L1 drifted outside safe cap |
