# Current Best Method

Updated: 2026-05-26

## Best Public Score

- Public score: **`0.73290`**
- Submission: `outputs/submissions/next_step41_specter30_from_scincl_ridge0_submission.csv`
- Archive: `outputs/submissions/next_step41_specter30_from_scincl_ridge0_submission.csv`
- Method artifacts: `outputs/step41_step13_backbone_retune/`
- Previous anchor: Step39 backbone-retuned BGE-M3 scored `0.73215`
- Jump over previous anchor: **`+0.00075`**

## Method Summary

Step41 SPECTER-up retuned the Step13 base semantic backbone inside the Step39 wrapper. The successful move kept E5 fixed at `0.50`, reduced SciNCL from `0.30` to `0.20`, and increased SPECTER2 from `0.20` to `0.30`, with Ridge kept at `0.00`. It then kept the Step39 calibrator and BGE-M3 wrapper unchanged.

Step41 winning base stack:

```text
base_stack = 0.50 * E5-large
           + 0.20 * SciNCL
           + 0.30 * SPECTER2
           + 0.00 * Ridge TF-IDF
```

Step41 winning wrapped blend:

```text
step41_backbone = 0.90 * base_stack + 0.10 * Ridge(alpha=10) Step18b targeted pattern calibrator

candidate = step13retune_e50p50_scincl0p20_specter0p30_ridge0p00_bge0p0275_l4p0
final_score = 0.9725 * step41_backbone + 0.0275 * BGE-M3 Ridge(alpha=3) diagnostic correction
threshold_lambda = 4.0
```

Expanded approximate component weights:

```text
final_score ≈ 0.437625 * E5-large
            + 0.175050 * SciNCL
            + 0.262575 * SPECTER2
            + 0.097250 * Step18b targeted calibrator
            + 0.027500 * BGE-M3 diagnostic correction
            + 0.000000 * Ridge TF-IDF
```

## Why this is the current best

1. **SPECTER2 was the protected base-stack weight.** The failed Step41 probes all reduced SPECTER2 to `0.10–0.15` and collapsed public to `0.716–0.721`.
2. **The winning direction transfers weight from SciNCL to SPECTER2, not from E5.** `E5=0.45/SciNCL=0.30/SPECTER=0.25` regressed to `0.73022`; `E5=0.50/SciNCL=0.20/SPECTER=0.30` became the new best.
3. **The Step39 wrapper remains useful.** The successful candidate kept the same `10%` targeted calibrator and `2.75%` BGE-M3 diagnostic correction.
4. **This is still a risky public-probe family.** The new best changes 16 rows vs Step39 and has `test_L1=0.155619`, so future work should inspect changed rows before broader sweeps.

## OOF + test metrics

```text
OOF QWK (constrained tuner) = 0.663141
OOF lift vs Step39         = +0.000555
OOF lift vs Step36         = +0.001043
OOF lift vs Step25         = +0.001881
Test combined L1 vs train  = 0.155619
Diff vs Step39 total       = 16
Diff vs Step36 total       = 17
Diff vs Step25 total       = 16
Public LB                  = 0.73290
```

Previous Step39 baseline:

```text
OOF QWK (constrained tuner) = 0.662586
Test combined L1 vs train  = 0.152263
Diff vs Step36 total       = 5
Diff vs Step25 total       = 8
Public LB                  = 0.73215
```

Recent neighborhood probes:

| Probe | OOF QWK | test_L1 | Diff vs anchor | Public |
| --- | ---: | ---: | ---: | ---: |
| Step39 `base 50/30/20 + calibrator 0.10 + BGE 0.0275` | 0.662586 | 0.152263 | 5 vs Step36 | 0.73215 |
| Step40 `calibrator bw=0.0975,l4` | 0.662409 | 0.152263 | 2 vs Step39 | 0.73079 |
| Step41 `E5=0.55/SciNCL=0.35/SPECTER=0.10/Ridge=0` | 0.659013 | 0.162331 | 12 vs Step39 | 0.72062 |
| Step41 `E5=0.60/SciNCL=0.30/SPECTER=0.10/Ridge=0` | 0.659128 | 0.169042 | 20 vs Step39 | 0.71643 |
| Step41 `E5=0.55/SciNCL=0.30/SPECTER=0.15/Ridge=0` | 0.659008 | 0.182465 | 18 vs Step39 | 0.71719 |
| Step41 `E5=0.45/SciNCL=0.30/SPECTER=0.25/Ridge=0` | 0.659475 | 0.172398 | 19 vs Step39 | 0.73022 |
| Step41 `E5=0.50/SciNCL=0.20/SPECTER=0.30/Ridge=0` | 0.663141 | 0.155619 | 16 vs Step39 | **0.73290** |

## Selected blend

| Component | Effective weight | Role |
| --- | ---: | --- |
| E5-large | 0.437625 | strongest safe encoder anchor, kept at 50% inside base stack |
| SciNCL | 0.175050 | scientific contrastive signal, reduced from Step39 base |
| SPECTER2 | 0.262575 | protected paper-similarity signal; increasing it to 30% inside base stack produced the new best |
| Step18b Ridge(alpha=10) targeted calibrator | 0.097250 | OOF-audited pattern/metadata correction |
| Step36 BGE-M3 Ridge(alpha=3) diagnostic correction | 0.027500 | low-weight frozen encoder signal for near-threshold rows |
| Ridge TF-IDF | 0.000000 | retained as reference, not used in base stack |

## Reproduce

```bash
python src/step41_step13_backbone_retune.py
```

Expected primary output:

```text
outputs/submissions/next_step41_specter30_from_scincl_ridge0_submission.csv
outputs/step41_step13_backbone_retune/summary.json
```

## Anchors kept for future stacking

| Folder | Public LB | OOF QWK | Role |
| --- | ---: | ---: | --- |
| `outputs/0.62820/` | 0.62820 | 0.5921 | lexical Ridge TF-IDF baseline |
| `outputs/0.69972/` | 0.69972 | 0.6389 | SPECTER2 fine-tune |
| `outputs/scincl_finetune/` | not submitted | 0.6269 | SciNCL fine-tune |
| `outputs/0.72103/` | 0.72103 | 0.6412 | earlier SciNCL/SPECTER2/Ridge stack |
| `outputs/0.72394/` | 0.72394 | 0.6593 | Step13 E5 4-anchor stack |
| `outputs/0.72808/` | 0.72808 | 0.6602 | Step18b targeted calibration |
| `outputs/bge_m3_frozen_anchor/` | 0.73054 | 0.6613 | Step25 BGE-M3 frozen diagnostic anchor |
| `outputs/step36_bge_m3_fine_sweep/` | 0.73213 | 0.6621 | Step36 BGE-M3 fine-sweep diagnostic anchor |
| `outputs/step39_step18b_backbone_retune/` | 0.73215 | 0.6626 | Step39 backbone-retuned BGE-M3 predecessor |
| `outputs/step41_step13_backbone_retune/` | **0.73290** | **0.6631** | **current best SPECTER-up Step39-wrapper anchor** |
| `outputs/qwen_lora_finetune_fixed_outputs/` | not submitted alone | 0.6205 | fixed Qwen LoRA anchor, useful as negative/shift reference |
| `outputs/scholarly_graph_anchor/` | not submitted | 0.3633 | metadata/graph anchor, diverse but too weak |

## Current decision rule

Use the Step41 SPECTER-up submission as the default public-best baseline. New candidates should not replace it unless they either:

- beat `0.73290` publicly, or
- improve local OOF while staying near the Step41 distribution band and preserving the validated direction: keep E5 around `0.50`, keep Ridge at `0`, and trade SciNCL against SPECTER rather than reducing SPECTER.

Next best search direction:

```text
E5 fixed near 0.50
Ridge fixed at 0.00
SPECTER sweep: 0.26–0.34
SciNCL sweep: 0.24–0.16
Keep Step39 calibrator weight = 0.10
Keep BGE-M3 weight = 0.0275
Keep threshold_lambda = 4.0 unless changed-row evidence says otherwise
```
