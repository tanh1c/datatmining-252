# Lessons learned — for the next agent / future-me

Living document. Each entry records a concrete observation we paid for in

## L27 — Step41 SPECTER-up retune: SPECTER was the protected backbone weight, not E5/SciNCL

**Evidence (2026-05-26):** After Step41 high-E5/SciNCL probes regressed badly, the user noticed all failed variants had reduced SPECTER from the Step39 base mix. Testing the opposite direction produced a new public best.

| Submission | Base mix E5/SciNCL/SPECTER/Ridge | Local OOF | test_L1 | Diff vs Step39 | Public | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Step39 anchor | 50 / 30 / 20 / 0 | 0.662586 | 0.152263 | — | 0.73215 | previous best |
| Step41 high E5/SciNCL | 55 / 35 / 10 / 0 | 0.659013 | 0.162331 | 12 | 0.72062 | severe regression |
| Step41 higher E5 | 60 / 30 / 10 / 0 | 0.659128 | 0.169042 | 20 | 0.71643 | severe regression |
| Step41 balanced high E5 | 55 / 30 / 15 / 0 | 0.659008 | 0.182465 | 18 | 0.71719 | severe regression |
| Step41 SPECTER25 from E5 | 45 / 30 / 25 / 0 | 0.659475 | 0.172398 | 19 | 0.73022 | mild regression |
| Step41 SPECTER30 from SciNCL | 50 / 20 / 30 / 0 | **0.663141** | 0.155619 | 16 | **0.73290** | **new best** |

**What worked:** do not reduce SPECTER. The successful move kept E5 at `0.50`, reduced SciNCL from `0.30` to `0.20`, and increased SPECTER from `0.20` to `0.30`, then kept the same Step39 wrapper: `90%` base stack + `10%` Ridge(alpha=10) targeted calibrator, followed by `97.25%` wrapped backbone + `2.75%` BGE-M3 Ridge(alpha=3), `threshold_lambda=4`.

**What failed:** increasing E5/SciNCL by cutting SPECTER to `0.10–0.15` destroyed public transfer. Increasing SPECTER to `0.25` by taking weight from E5 also did not help. The public-positive direction specifically appears to be **shift weight from SciNCL to SPECTER while preserving E5 at 50%**.

**Rule:** inside the Step39/41 best family, SPECTER is a protected semantic anchor and should not be reduced below `0.20`. The next sweep should be a narrow SciNCL→SPECTER transfer around `E5=0.50, Ridge=0`, e.g. SPECTER `0.26–0.34`, SciNCL `0.24–0.16`, with the Step39 calibrator+BGE wrapper fixed.

**Action items.**
- [x] Promote `next_step41_specter30_from_scincl_ridge0_submission.csv` as current best public anchor (`0.73290`).
- [x] Record `next_step41_specter25_from_e5_ridge0_submission.csv` as a regression (`0.73022`).
- [ ] Inspect the 16 Step41-vs-Step39 changed rows to understand why the SPECTER30/SciNCL20 direction transferred.
- [ ] If more probing is allowed, sweep only the SciNCL→SPECTER tradeoff around the new best, not E5 increases.

---

## L26 — Step40/41 early probes: fine calibrator sweeps and SPECTER-reducing large-backbone retunes regress

**Evidence (2026-05-25):** Step40 and the first Step41 probes tested two plausible follow-ups after Step39 became the public best at `0.73215`: fine-sweeping the Step18b calibrator and retuning the large Step13 E5/SciNCL/SPECTER backbone by increasing E5/SciNCL. These regressed publicly.

| Submission | Local OOF | Diff vs Step39 | Public | Result |
| --- | ---: | ---: | ---: | --- |
| Step39 anchor `next_step39_best_backbone_oof_submission.csv` | **0.662586** | — | **0.73215** | previous best |
| Step40 closest fine probe `bw=0.0975,l4` | 0.662409 | 2 | 0.73079 | regressed despite tiny diff |
| Step41 `E5=0.55/SciNCL=0.35/SPECTER=0.10/Ridge=0` | 0.659013 | 12 | 0.72062 | severe regression |
| Step41 `E5=0.60/SciNCL=0.30/SPECTER=0.10/Ridge=0` | 0.659128 | 20 | 0.71643 | severe regression |
| Step41 `E5=0.55/SciNCL=0.30/SPECTER=0.15/Ridge=0` | 0.659008 | 18 | 0.71719 | severe regression |

**What failed:** even the Step40 two-row probe lost `-0.00136` public QWK, so tiny diff from Step39 is not automatically safe. The early Step41 high-E5/SciNCL variants also showed that reducing SPECTER away from the Step13/39 base mix is very harmful, even with the same Step39 calibrator and BGE-M3 wrapper.

**Rule:** do not continue fine calibrator sweeps or E5/SciNCL-increase sweeps that cut SPECTER. Step39 is no longer the best, but its wrapper remains useful; the validated new direction is L27's SciNCL→SPECTER transfer.

**Action items.**
- [x] Record Step40 public regression (`0.73079`).
- [x] Record Step41 forced high-E5/SciNCL regressions (`0.72062`, `0.71643`, `0.71719`).
- [x] Revise this lesson after L27 showed SPECTER-up from SciNCL is the exception that improves public.

---

## L25 — Step39 backbone retune: increasing Step18b targeted calibrator to 10% barely beats Step36

**Evidence (2026-05-25):** Step39 `next_step39_best_backbone_oof_submission.csv` scored public `0.73215`, beating Step36 `0.73213` by a tiny `+0.00002`.

| Submission | Local OOF | Lift vs Step36 | test_L1 | Diff vs Step36 | Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step36 `Step18b w=0.08 + BGE w=0.0275,l4` | 0.662098 | baseline | 0.148908 | — | 0.73213 |
| Step39 `Step18b calibrator w=0.10 + BGE w=0.0275,l4` | **0.662586** | **+0.000488** | 0.152263 | 5 | **0.73215** |

**What worked:** after Step37/Step38 showed the BGE-M3 `2.75%` correction itself was brittle from both sides, retuning the much larger Step18b backbone was the right next knob. Raising the targeted Ridge(alpha=10) calibrator inside Step18b from `0.08` to `0.10` increased OOF and transferred by a very small public margin while keeping the proven BGE-M3 diagnostic fixed.

**What failed / caution:** the public lift is only `+0.00002`, effectively a knife-edge improvement. The candidate has higher `test_L1=0.152263` than Step36 and changes 5 rows vs Step36, so do not generalize this into broad backbone-weight sweeps without changed-row inspection. It says the Step18b audit features still have a little useful signal, not that heavier calibrator weight is broadly safe.

**Rule:** when BGE-M3 micro-sweeps around Step36 fail, the next useful knob is the Step18b targeted-calibrator backbone, but only in tiny increments and with public-risk labeling. Treat Step39 `backbone_weight=0.10, BGE w=0.0275, lambda=4` as the new public anchor until beaten; future attempts should inspect the 5 Step39-vs-Step36 changed rows before trying `0.09/0.10/0.11` micro-neighborhoods.

**Action items.**
- [x] Promote Step39 `backbone_weight=0.10, BGE w=0.0275, lambda=4` as current best public anchor (`0.73215`).
- [ ] Inspect the 5 changed rows between Step36 and Step39; they are now the key forensic examples.
- [ ] If more probing is allowed, search around Step18b backbone `0.09–0.105` only after changed-row review, not as a blind broad sweep.

---

## L24 — Step36 BGE-M3 fine sweep: small-diff risky probe beat Step25, while higher-OOF probes split

**Evidence (2026-05-24):** Step36 `next_step36_risky_small_diff_ridge_a3_w0p0275_l4_submission.csv` scored public `0.73213`, beating Step25 `0.73054` by `+0.00159`.

| Submission | Local OOF | Lift vs Step25 | test_L1 | Diff vs Step25 | Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step25 BGE-M3 safe | 0.661261 | baseline | 0.145552 | — | 0.73054 |
| Step36 `ridge_a3`, `w=0.0300`, `lambda=4` | **0.663105** | **+0.001845** | 0.145552 | 9 | 0.72602 |
| Step36 `ridge_a3`, `w=0.0275`, `lambda=4` | 0.662098 | +0.000837 | 0.148908 | **5** | **0.73213** |
| Step36 `ridge_a3`, `w=0.0375`, `lambda=4` | **0.663312** | **+0.002051** | 0.148908 | 11 | 0.73201 |
| Step37 `ridge_a3`, `w=0.0280`, `lambda=6` | 0.662340 | +0.001079 | 0.148908 | 4 vs Step25 / 1 vs Step36 | 0.72879 |
| Step37 `ridge_a3`, `w=0.0290`, `lambda=6` | **0.663513** | **+0.002253** | 0.145552 | 6 vs Step25 / 5 vs Step36 | 0.72916 |
| Step38 `ridge_a3`, `w=0.0265`, `lambda=4` | 0.662763 | +0.001503 | 0.148908 | 4 vs Step25 / 5 vs Step36 | 0.72635 |
| Step38 `ridge_a3`, `w=0.0260`, `lambda=6` | 0.662213 | +0.000953 | 0.148908 | 4 vs Step25 / 1 vs Step36 | 0.72879 |

**What worked:** stay very close to the validated Step25 mechanism, but retune the BGE-M3 correction from `ridge_a30, w=0.02` to `ridge_a3, w=0.0275` with `threshold_lambda=4.0`. This kept Step18b as the backbone and used BGE-M3 only as a near-threshold diagnostic.

**What failed:** OOF ranking alone was misleading. The `w=0.0300` probe had much higher OOF and the same `test_L1` as Step25, but public regressed to `0.72602`. The highest-OOF Step36 probe `w=0.0375` scored well (`0.73201`) but still lost to the smaller-diff `w=0.0275` probe. Step37 then showed that even a one-row-different micro-retune (`w=0.0280,l6`) regressed to `0.72879`, and the best local OOF micro-retune (`w=0.0290,l6`, OOF `0.663513`) regressed to `0.72916`. Step38 tested the lower-weight side (`w=0.0265,l4` and `w=0.0260,l6`); both had positive OOF lift over Step36/Step25 but public regressed to `0.72635` and `0.72879`, so lowering BGE-M3 weight does not repair the brittle changed-row directions.

**What this revises:** the strict `test_L1 <= 0.147` cap is a good default safety rule, not an absolute law. When a candidate is extremely close to a public-validated anchor and changes only a handful of rows, an explicitly labeled risky public probe can be justified. The user's pushback to test Step36 was correct; without that probe, the new best would have been missed.

**Rule:** near a public-best anchor, rank candidates by a combination of OOF, distribution safety, and **diff versus the current best**, but after Step37/Step38 do not assume that tiny diff is automatically safe. Step36 `w=0.0275,l4` is a sharp public optimum from both sides: increasing to `w=0.028-0.029` and decreasing to `w=0.026-0.0265` both regressed, including one-row-different probes. Further attempts should inspect the actual changed rows before submitting, not continue blind weight/threshold micro-sweeps.

**Action items.**
- [x] Promote Step36 `w=0.0275` as current best public anchor (`0.73213`).
- [x] Record the failed/high-OOF Step36 probes so future sweeps do not over-trust OOF.
- [ ] Inspect the 5 changed rows between Step25 and Step36 winner; these are now the most valuable forensic examples.
- [x] Test tiny single-knob neighborhoods around `ridge_a3, w=0.0275, lambda=4`; Step37 and Step38 regressed on both higher and lower weights, so stop blind micro-sweeps.

---

## L23 — Step25 BGE-M3 safe frozen anchor became the new best; small distribution-safe encoder diagnostics can still transfer

**Evidence (2026-05-24):** Step25 `next_step25_bge_m3_best_safe_submission.csv` scored public `0.73054`, beating Step18b `0.72808` by `+0.00246`.

| Submission | Local OOF | Lift vs Step18b | test_L1 | Changed test rows | Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step18b | 0.660176 | baseline | 0.145552 | — | 0.72808 |
| Step25 BGE-M3 | **0.661261** | **+0.001084** | **0.145552** | **9** | **0.73054** |

**What worked:** BGE-M3 was used as a frozen diagnostic anchor with a tiny Ridge blend (`ridge_a30`, `w=0.02`) over Step18b, not as a broad dominant model. It preserved the public-validated distribution band and only changed a few rows.

**What this revises:** Step31's reverse-forensics warned that most changes away from Step18b hurt public, but Step25 proves a carefully gated low-weight frozen encoder can still improve. The mistake would be to generalize this into heavier BGE-M3 sweeps; the winning pattern is narrow and distribution-safe.

**Rule:** after Step18b, the next improvement path is not larger generative LMs or broad retraining; it is small frozen-anchor diagnostic corrections with `test_L1 <= 0.147`, changed rows controlled, and manual inspection of the actual changed rows.

**Action items.**
- [x] Promote Step25 as current best public anchor (`0.73054`).
- [ ] Inspect Step25's 9 changed test rows vs Step18b and categorize them.
- [ ] Use Step25 as the new baseline for future micro-gates.
- [ ] Do not submit Step26/28/30 broad candidates; they failed safety transfer.

---

## L22 — Surgical Qwen14B disagreement gates can preserve public, but OOF-perfect tiny gates may only tie the best

**Evidence (2026-05-22):** After Step22 broad Qwen14B blending regressed public, Step23 used clean Qwen14B only as a narrow disagreement signal over the public-best Step18b score. The winning gate kept Step18b thresholds fixed and changed only rows near the Step18b 3/4 boundary where Qwen disagreed by at least 2 labels.

| Submission | Local OOF | Lift vs Step18b | test_L1 | Changed | OOF improved/worsened | Public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step18b current best | 0.660176 | baseline | 0.145552 | — | — | **0.72808** |
| `next_step23_best_safe_gate_submission.csv` | **0.663062** | **+0.002886** | 0.145552 | 12 | **12/0** | **0.72808** |

**What worked:** Unlike Step22, the surgical gate did not harm public. It preserved Step18b thresholds and distribution, changed only 12 rows, and avoided broad score blending. This confirms the right way to use Qwen14B is as a row-level diagnostic, not as a global stack weight.

**What did not work:** Even a locally perfect 12/12 OOF gate did not improve public beyond Step18b. The likely explanation is public/private split mismatch at the individual-row level: the OOF examples were real train corrections, but the analogous test rows were either absent from public, already correctly handled by Step18b, or balanced by hidden misses.

**Rule:** fixed-threshold small gates are safer than per-rule threshold retuning for this stage. They isolate the effect of the changed rows and preserve the public-proven Step18b calibration. Per-rule differential-evolution threshold tuning across many gates would inflate OOF and make it unclear whether lift comes from the gate or from retuning the whole label distribution.

**Action items.**
- [x] Record Step23 public tie (`0.72808`) in `outputs/leaderboard_tracking.md`.
- [x] Keep Step18b as the main best method; Step23 is at most a private-test hedge because it ties public rather than beating it.
- [ ] Do not submit `second_safe_gate`; it is effectively duplicate signal and unlikely to add useful leaderboard information.
- [ ] Future attempts should not search broader Qwen gates. Instead, inspect exactly why the 12 Step23 OOF fixes did not move public, then add only non-Qwen pattern features or constraints that target unmatched public/private failure modes.

---

## L21 — Clean Qwen14B proves the adapter fix but still fails public; high-capacity LLM anchors need changed-row validation beyond test_L1

**Evidence (2026-05-21):** Step 22 reran Qwen2.5-14B-Instruct LoRA on Modal H200 using the fixed fresh-base-per-fold discipline from Step 14:

```text
model = Qwen/Qwen2.5-14B-Instruct
precision = bf16 native
LoRA = r32 / alpha64
max_len = 1024
batch = 8 x grad_accum 2
folds = 5 x 1 seed
```

The clean single anchor was real and no longer showed the old adapter-chain failure:

| Anchor | OOF QWK | Fold round-QWK spread | Notes |
| --- | ---: | --- | --- |
| clean Qwen14B | 0.650885 | 0.6212-0.6694 | fresh base loaded per fold; no PEFT adapter-chain warning |
| E5 reference | 0.6496 | — | encoder anchor reference |

Stacking looked excellent locally:

| Submission | weights q/e5/sc/sp/r | OOF | lift vs Step18b | test_L1 | changed | improved/worsened | Public |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step18b current best | targeted E5 calibrator | 0.660176 | baseline | 0.145552 | — | — | **0.72808** |
| Step22 main | .15/.3825/.2975/.17/0 | **0.668669** | **+0.008492** | 0.145552 | 137 | 87/50 | **0.72403** |
| Step22 safer hedge | .10/.495/.225/.18/0 | 0.667131 | +0.006954 | 0.144990 | 108 | 71/37 | **0.72106** |

**Conclusion:** the old Qwen14B failure was not only adapter-chain leakage. Clean Qwen14B is a strong and diverse anchor (`r≈0.88-0.89` vs encoders), but the current train/public split still penalizes its changed-row directions. The safe-band rule (`test_L1 <= 0.147`) is necessary but no longer sufficient for high-capacity generative-LM anchors: both Step22 submissions stayed inside the band and still regressed.

**What failed:** local OOF rewarded many Qwen-driven corrections, but public did not. The main candidate had 54 risky promotions and 3 core-ish 5→4 demotions; the hedge reduced this to 37 and 1 but public became even worse. Therefore simply lowering Qwen weight is not enough.

**Rule:** do not submit more clean-Qwen14B weight sweeps unless there is a new constraint that directly targets public-failing changed-row directions. For high-capacity LLM anchors, require all of:
1. meaningful OOF lift,
2. safe `test_L1`,
3. changed-row forensic profile that is not dominated by KR/CAV/formal-adjacent promotions or core-ASP demotions,
4. preferably validation from a second seed or an orthogonal audit, not one 5-fold seed alone.

**Action items.**
- [x] Record `next_step22_qwen14b_clean_main_high_oof_safe_l1_submission.csv` public `0.72403` as a regression vs Step18b.
- [x] Record `next_step22_qwen14b_clean_safer_hedge_submission.csv` public `0.72106` as a stronger regression.
- [x] Keep Step18b (`0.72808`) as current best.
- [ ] Stop Qwen14B weight-only sweeps; if revisiting Qwen, use it only as a diagnostic feature with stricter row-level gates or pairwise reranking, not as a broad blend weight.
- [ ] Future 0.75 attempts should return to OOF-error targeted calibration / row-level forensic constraints rather than adding more broad high-capacity anchors.

---

## L20 — Tiny OOF-audited pattern calibration can beat the encoder ceiling if it stays distribution-safe

**Evidence (2026-05-21):** Step 18 audited the current-best E5 stack OOF errors and found systematic label-axis mistakes that pure encoder similarity was not correcting:

| Error group | n | mean true | mean pred | Key pattern |
| --- | ---: | ---: | ---: | --- |
| false-low big (`pred <= true-2`) | 200 | 3.835 | 1.625 | KR/CAV/LICS/formal papers often lack explicit ASP tokens |
| false-high big (`pred >= true+2`) | 239 | 1.548 | 3.837 | ASP/logic/AI keywords can over-trigger high labels |
| true label 5 predicted <=3 | 49 | 5.000 | 2.633 | many are KR/AI/formal high-relevance papers without core ASP wording |
| true label 1 predicted >=3 | 135 | 1.000 | 3.385 | explainable AI / logic-programming / ASP-control titles can be low-label traps |

Step 18b then trained a tiny OOF-safe calibration layer over the public-proven E5 stack:

```text
base_score = 0.50 * e5 + 0.30 * scincl + 0.20 * specter2 + 0.00 * ridge
calibrator = Ridge(alpha=10) over base_score + OOF-audit pattern/metadata features
final_score = 0.92 * base_score + 0.08 * calibrator
```

**Submission result:**

| Submission | OOF | round-QWK | test_L1 | Public | Δ vs E5 best 0.72394 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E5 safest anchor | 0.659285 | **0.647483** | **0.142196** | 0.72394 | baseline |
| `next_step18b_targeted_blend_ridge_a10_w0p08_submission.csv` | **0.660176** | 0.646028 | 0.145552 | **0.72808** | **+0.00414** |

The OOF lift was only +0.00089 and round-QWK dropped slightly, but the public lift was +0.00414. This is the first successful break above the E5 encoder ceiling after Qwen/metadata experiments failed.

**Why this transferred when Qwen did not:**
- The calibrator is small (`w=0.08`) and regularized, so it cannot dominate the proven E5 score.
- Features came from train OOF error structure, not public-row tuning.
- `test_L1=0.1456` stayed within the L17 safe cap (`<=0.147`).
- The new signal is label-axis semantics (meta/proceedings, keyword traps, KR/formal false-lows), not another same-family encoder.

**Rule:** once a strong encoder stack reaches a ceiling, small OOF-audited feature calibration is a better next move than adding noisy high-capacity anchors. Require all three before promotion: (1) private-safe features only, (2) tiny blend weight / strong regularization, (3) `test_L1` inside the current public-validated safe band.

**Action items.**
- [x] Promote `next_step18b_targeted_blend_ridge_a10_w0p08_submission.csv` as the current best public anchor (`0.72808`).
- [x] Archive artifacts under `outputs/0.72808/` and update `outputs/current_best_method.md`.
- [ ] Do not submit heavier Step 18b tree blends unless they are reworked to stay inside `test_L1 <= 0.147`; the best tree blend had higher OOF but unsafe distribution.
- [ ] Next improvement should deepen the same approach: inspect changed rows and add only a few targeted, OOF-validated corrections rather than increasing calibrator capacity.

---


## L19 — Qwen14B OOF lift failed public even inside the L17 safe band; adapter-chain LLM OOF is not trustworthy

**Evidence (2026-05-20):** Qwen2.5-14B LoRA single-anchor output looked much stronger than fixed Qwen7B:

| Anchor | OOF QWK | fold round-QWK spread | test_L1 | Notes |
| --- | ---: | ---: | ---: | --- |
| fixed Qwen7B | 0.6205 | 0.5038–0.6546 | ~0.19 single | fresh-base fix applied |
| Qwen14B old-notebook run | **0.6561** | 0.6133–0.6689 | **0.1925** | PEFT adapter-chaining warnings present |

The raw Qwen14B anchor was distribution-unsafe, so Step 17b/17c tried conservative E5/SciNCL/SPECTER stacks. Local metrics looked promising:

| Submission | weights q/e5/sc/sp/r | OOF | round-QWK | test_L1 | Public | Δ vs E5 best 0.72394 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E5 safest anchor | 0/.50/.30/.20/0 | 0.6593 | 0.6475 | 0.1422 | **0.72394** | baseline |
| Qwen14B safe-heavy | .20/.40/.25/.15/0 | **0.6681** | **0.6541** | **0.1456** | **0.71453** | **−0.00941** |
| Qwen14B risky-high | .35/.26/.195/.13/.065 | **0.6760** | 0.6527 | 0.1718 | **0.71408** | **−0.00986** |

This is the strongest negative transfer so far: the safe-heavy candidate obeyed the L17 hard cap (`test_L1 <= 0.147`) and had a large local OOF lift (+0.0088), yet public dropped almost one full point of QWK. Therefore L17's safe-band rule is necessary but not sufficient when the new anchor's OOF is contaminated or miscalibrated.

**Most likely cause:** the Qwen14B notebook used the old PEFT flow and emitted adapter-chaining warnings from fold 2 onward (`Already found a peft_config attribute... multiple adapters`). The resulting OOF can look strong because folds are not independent clean fits. The public split exposes the leakage/miscalibration.

**Rule:** do not promote LLM LoRA anchors trained with adapter-chaining warnings, regardless of local OOF or blend test_L1. For Qwen/Llama-style LoRA, only trust runs that reload a fresh base model per fold and clean up the model after each fold.

**Action items.**
- [x] Mark both Qwen14B submissions as public regressions in `outputs/leaderboard_tracking.md`.
- [x] Keep `outputs/qwen14b_lora_finetune_outputs/` and `outputs/step17c_qwen14b_heavy_5anchor_stack/` as negative references, not promotion candidates.
- [ ] If revisiting Qwen14B, rerun with the fixed fresh-base-per-fold notebook before any further submission.
- [ ] Prioritize the pairwise/cross-encoder reranker path over more adapter-chained Qwen weight sweeps.

---


## L18 — Metadata/graph diversity without label signal is not enough to break the E5 ceiling

**Evidence (2026-05-20):** Step 16 built a standalone scholarly graph/concept metadata anchor from cached OpenAlex/Semantic Scholar/Crossref/OpenCitations features. The best candidate was ExtraTrees over citation/reference/FWCI/source-agreement/venue/year/author/S2-field features:

| Anchor | OOF QWK | round-QWK | test_L1 | r vs E5 stack |
| --- | ---: | ---: | ---: | ---: |
| Scholarly graph ExtraTrees | 0.3633 | 0.2591 | 0.2395 | 0.507 |
| E5 safest stack | 0.6593 | 0.6475 | 0.1422 | 1.000 |

Diversity was real (r≈0.51 vs E5 stack), but signal was far below the L11 floor. Blending confirmed it cannot help the current public-validated anchor:

| Blend | OOF | round-QWK | test_L1 | Verdict |
| --- | ---: | ---: | ---: | --- |
| graph 0.03 / E5 0.97 | 0.6598 | 0.6468 | 0.1523 | OOF tiny +, outside L17 safe band |
| graph 0.05 / E5 0.95 | 0.6591 | 0.6460 | 0.1456 | inside L17 safe band, but OOF below E5 |
| graph 0.08 / E5 0.92 | 0.6616 | 0.6398 | 0.1892 | OOF +, distribution unsafe |

**Rule:** metadata/graph anchors need both diversity and real label signal. Citation counts, FWCI, source agreement, venue/year percentiles, and broad concept fields mostly measure paper impact/indexing context, not ASP relevance. If standalone OOF is ~0.36, even r≈0.5 diversity is unusable except as a diagnostic.

**Action items.**
- [x] Do not submit `next_e5_scholarly_graph_*`; no candidate both beats E5 and stays inside test_L1 ≤ 0.147.
- [x] Keep `outputs/scholarly_graph_anchor/` as a negative reference for metadata-only graph features.
- [ ] If revisiting external metadata, it must add **semantic** concept text/abstract-like features (e.g. OpenAlex concepts/topics as text into an encoder), not raw citation graph metrics alone.

---


## L17 — Out-of-family signal that passes blend probe can still regress public when OOF→public discount goes NEGATIVE

**Evidence (2026-05-19, public score 2026-05-20):** Qwen2.5-7B LoRA fine-tune (5 folds × 1 seed, LoRA r=32, bf16 native on H200, verbalizer inference) was the **first non-encoder anchor to clear both L11/L12 floors**:

| Anchor | OOF | r vs E5 | r vs BGE | Family |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B LoRA | 0.6326 | **0.888** | 0.883 | generative LM (out-of-family) |
| E5-large-v2 | 0.6417 | 1.000 | 0.944 | contrastive sentence sim |
| BGE-large | 0.6228 | 0.944 | 1.000 | contrastive sentence sim |
| SPECTER2 | 0.6297 | 0.916 | 0.909 | citation contrastive |
| SciNCL | 0.6117 | 0.909 | 0.911 | citation contrastive |

Diversity floor cleared (r 0.88 — first non-Ridge anchor in the 0.85-0.92 sweet spot). Signal floor cleared (0.63 ~ SPECTER2). Blend probe found a **first-ever 5-anchor lift over 4-anchor**: best 5-anchor mix had round-QWK 0.6543 vs 4-anchor safest_e5 at 0.6475 — a +0.0068 OOF lift, the largest stack lift since BGE entered.

**Submission result (2026-05-20):**

| Submission | weights (q/e5/sc/sp/r) | OOF | test_L1 | Public | Δ vs safest_e5 (0.72394) |
| --- | --- | ---: | ---: | ---: | ---: |
| safest_e5 (anchor) | 0/.50/.30/.20/0 | 0.6593 | **0.1422** | **0.72394** | (baseline) |
| **safest_with_qwen** | .20/.40/.25/.15/0 | 0.6619 | 0.1523 | **0.72184** | **OOF +0.0026, L1 +0.010, public −0.00210** |

**For the first time in the project, OOF lift translated to a public regression.** Previous patterns (L13):
- BGE 4-anchor (vs 0.72103): OOF +0.0155 → public +0.0027 (6× discount, positive)
- E5 4-anchor (vs BGE): OOF +0.0026 → public +0.0002 (13× discount, positive)
- **Qwen 5-anchor (vs E5): OOF +0.0026 → public −0.00210 (NEGATIVE)**

**Three mutually-non-exclusive hypotheses:**

1. **PEFT adapter chaining bug.** Cell 13 of step 14 calls `get_peft_model(base_model, ...)` per fold without proper teardown. Folds 2-5 emitted `UserWarning: Already found a peft_config attribute in the model. This will lead to having multiple adapters in the model.` Fold 1 OOF QWK was 0.649 (highest); folds 2-5 averaged 0.624. Suggests fold-1 adapter weights leak into fold-2-5 training, inflating overall OOF without translating to test. Most plausible because it has **direct evidence** (peft warning).

2. **Verbalizer score distribution mis-aligned with public split.** Qwen pred dist on test (after blending): `{1:123, 2:60, 3:54, 4:38, 5:23}`. Train dist: `{1:123, 2:70, 3:50, 4:35, 5:22}` — label 2 under-predicted by 10 rows (~14% deficit), label 4 over-predicted by 3 rows. The constrained tuner forces OOF distribution to match train, but the **test score distribution** can still drift if the model's confidence calibration differs systematically between train and test populations.

3. **Single-seed instability.** Qwen used 1 seed × 5 folds = 5 models. Encoder anchors used 3 seeds × 5 folds = 15 models. With only 5 models, the test-side score variance is ~√3 ≈ 1.7× higher than encoders. Some of this variance can flip the sign of a small OOF lift on public.

**Cross-cutting pattern (now 17 lessons in):**

The OOF→public discount has a sign:
- L13 noted the magnitude was widening (6× → 13×) as we approached the encoder family ceiling.
- L17 adds: **the sign can flip when a candidate's predicted distribution moves further from train than the anchor's, even by ~0.01 in test L1 distance.** The L1 gap that BGE+E5 lived inside (~0.005-0.015) was the signal-translation zone; outside that zone, OOF gains do not transfer.

This is a **revision of L7's "rank by test L1 first"** rule. L7 said test_L1 is a *first-pass filter*. L17 says test_L1 is a **hard upper bound**: any candidate with test_L1 > anchor's test_L1 + ~0.01 is at risk of negative discount, regardless of OOF lift. The current public-validated anchor's test_L1 is 0.142; the safe band is roughly 0.142 ± 0.005 = [0.137, 0.147]. Anything outside is gamble.

**Rule of thumb (L17 — supersedes L7's filter rule).**

For follow-up submissions to a public-validated anchor:
1. **Hard cap test_L1 ≤ anchor's test_L1 + 0.005.** Anything above this is too far from the public-validated distribution; OOF lift can no longer compensate.
2. Within the hard cap, rank by OOF QWK. Submit the candidate that fits both constraints.
3. If no Qwen-bearing candidate fits, **the anchor is at a local optimum**. Stop sweeping weights; lift requires a different anchor or training fix, not a weight perturbation.

For step 15 specifically:
- Best Qwen-bearing test_L1 was 0.1523 (safest_with_qwen). Anchor test_L1 0.1422. Gap: 0.010 — outside the safe band by 2×. Submission was a gamble; gamble lost.
- All other Qwen-bearing candidates have test_L1 ≥ 0.155, even further out. **Do not submit them.**

**Post-fix result (2026-05-20):** fixed `notebooks/step14_qwen_lora_finetune.ipynb` to reload a fresh base model and attach a fresh LoRA adapter per fold. Clean Qwen 7B single-anchor result:

| Run | OOF QWK | OOF MAE | macro-F1 | Fold notes |
| --- | ---: | ---: | ---: | --- |
| Qwen 7B before PEFT fix | ~0.6326 | — | — | adapter-chaining warning present |
| **Qwen 7B after PEFT fix** | **0.6205** | 0.8176 | 0.3939 | fold spread 0.5038–0.6546 |

The fixed run dropped by ~0.012 OOF, confirming the previous Qwen signal was at least partly inflated. It still clears the old 0.55 floor, but is below SciNCL/BGE/SPECTER/E5 and has high fold variance.

Clean fixed-Qwen Step 15b blend sweep (`outputs/step15b_qwen_fixed_5anchor_stack/candidates.csv`) found the same public-risk pattern:

| Candidate | weights q/e5/sc/sp/r | OOF | round-QWK | test_L1 | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| E5 safest | 0/.50/.30/.20/0 | 0.6593 | 0.6475 | **0.1422** | current best |
| lowest-L1 fixed-Qwen | .15/.45/.25/.15/0 | 0.6646 | 0.6489 | 0.1523 | outside L17 safe band |
| high-OOF fixed-Qwen ≤0.165 | .20/.40/.20/.15/.05 | 0.6653 | 0.6472 | 0.1556 | outside L17 safe band |
| probe winner fixed-Qwen | .25/.35/.25/.15/0 | 0.6628 | 0.6501 | 0.1763 | unsafe |

No Qwen-bearing candidate stayed within the current safe cap (`test_L1 <= 0.147`). The only safe candidate is E5-only. Therefore the original public regression was not just a bad candidate choice; Qwen 7B clean remains distribution-unsafe for this stack.

**Action items.**
- [x] Mark Qwen 5-anchor `safest_with_qwen` as a public regression (0.72184) in `outputs/leaderboard_tracking.md`.
- [x] Fix the PEFT adapter chaining bug and import clean rerun outputs to `outputs/qwen_lora_finetune_fixed_outputs/`.
- [x] Rerun fixed-Qwen blend quantification; result: no Qwen-bearing candidate satisfies L17 safe band.
- [x] **DO NOT submit** `next_qwen_5anchor_{probe_winner,high_oof_5anchor}_submission.csv` or fixed-Qwen Step 15b candidates.
- [ ] Optional only if pursuing LLMs further: pivot to Qwen 14B / stronger LLM anchor. Do not spend 3-seed budget on Qwen 7B unless the goal is variance measurement, not leaderboard lift.
- [ ] Final-submission strategy (per private-test risk discussion): if no further lift is found, the 2 Kaggle final picks should be `safest_e5` (current public best, OOF 0.6593, test_L1 0.142) + `safest` BGE 4-anchor (different risk profile, ridge=0.05) — not two near-copies.

---


## L16 — "Frozen knob" must be re-tested isolated when anchor changes; bias mindset cleared by E5 max_len audit

**Backstory (2026-05-18):** comparison with a friend's SPECTER2 fine-tune (Public LB 0.71254 vs ours 0.69972, +0.013) showed his recipe used `MAX_LEN=384` while ours stayed at `256`. After our 0.68718 attempt (v3 cache + max_len 384 + unconstrained tuner) regressed, L6 codified "don't change multiple knobs at once" — but in practice we then **froze** `MAX_LEN=256` for ALL subsequent fine-tunes (SciNCL/SciBERT/BGE/E5) without ever re-isolating that single variable. That's L6 used as a freeze excuse.

**Audit experiment (step 12b):** clone of step 12 E5 recipe, only `MAX_LEN: 256 -> 384`. Everything else identical (mean pool, fp32 load, bf16 autocast, 5×3 folds, AdamW split LR, constrained tuner).

| Run | MAX_LEN | OOF QWK |
| --- | ---: | ---: |
| step 12 E5 (anchor) | 256 | **0.6496** |
| step 12b E5 ablation | 384 | 0.6433 |
| Δ | +128 tokens | **−0.0063** |

**Result:** the longer context **hurt** E5 by 0.0063 OOF QWK. The mindset bias was wrong; staying at 256 for E5 was correct.

**Why max_len=384 helps SPECTER2 but not E5/BGE (a pre-training data lesson):**
- SPECTER2 is pre-trained with a **citation-triplet contrastive objective** on **full scientific abstracts** (typically 200-500 tokens). 384 sits inside its training distribution.
- E5 and BGE are pre-trained with **sentence-pair contrastive objectives** on web/MS-MARCO data where query-passage pairs average ~150 tokens. 256 is already at the *upper edge* of the typical training context. 384 pushes E5 out-of-distribution → encoder behaviour gets noisier rather than richer.
- This means **max_len is not a universal knob; it interacts with the encoder's pre-training context distribution.** Friend's win wasn't "384 is better"; it was "384 fits SPECTER2's distribution".

**Mindset lesson (the one that mattered most).**

L6 says "change one knob at a time when debugging a regression". That rule is right. The trap was **using L6 to justify never re-testing a knob that had regressed in a multi-knob combo**. After 0.68718 (which moved 3 knobs together), the correct response was:
- "We don't know which of the 3 caused the regression." ✓ (L6 catches this)
- "Therefore freeze all 3 forever." ✗ (this is the bias)
- "Therefore re-test each individually when conditions change." ✓ (this is L16)

A frozen knob from a previous combo is a hypothesis, not a fact. When the anchor set, batch size, precision, or other knobs change (which happened multiple times across step 6 → step 13), the frozen knob should be re-isolated **at least once** before being declared safe.

**Rule of thumb (refines L6 + L10 + L14 with a re-test escape valve).**

- After a multi-knob regression, freeze knobs only **as a hypothesis**. Re-test each knob *isolated* the next time the surrounding configuration changes meaningfully.
- For knobs that interact with pre-training distribution (max_len, tokenizer prefix, pooling), the re-test should happen **per encoder family**, not per encoder. Once max_len=256 was confirmed for one contrastive-sentence encoder (E5), it's reasonable to extrapolate to BGE/MS-MARCO siblings without re-testing each.
- Audit the freeze list explicitly when adding a new anchor: if any frozen knob has not been re-tested under the new conditions, that's a debt to address before the new anchor is committed.

**Action items.**
- [x] E5 max_len=256 confirmed correct. Bias cleared.
- [x] Save `outputs/e5_large_finetune_max384/` as a negative reference.
- [ ] Do **not** re-run BGE/SciNCL/SciBERT with max_len=384 — same family as E5 (contrastive sentence similarity), same expected behaviour. Cost/benefit too low.
- [ ] **DO** consider one cheap re-run of SPECTER2 with the friend's full recipe (max_len=384 + batch=8 + constrained tuner + v2 cache) as a single SPECTER2 anchor variant — could lift the SPECTER2 contribution to the stack. Time budget ~25 min on H200. Defer until after Qwen LoRA result.
- [ ] When introducing the Qwen LoRA anchor (step 14), audit its frozen knobs explicitly. Currently MAX_LEN=1024 (H200 path) was a config choice, not a frozen knob — no audit needed. The 4-bit/bf16 split is also a hardware choice, not a quality knob.

---


## L15 — Same-family encoders are not additive; pick the best one and replace, do not stack

**Evidence (2026-05-18):** fine-tuned `intfloat/e5-large-v2` (335M, BERT-large arch, mean pool, bf16) using the exact same recipe as step 9 BGE plus the mandatory `"passage: "` prefix. E5 produced the **strongest single anchor** in the project (OOF round-QWK 0.6417 vs BGE 0.6228, SPECTER2 0.6297, SciNCL 0.6117) yet only added **+0.00020 public LB** (0.72374 → 0.72394).

**Floor checks (per L12):**

| Floor | Required | E5 | Pass? |
| --- | --- | --- | --- |
| Signal floor (OOF round-QWK ≥ 0.51) | yes | 0.6417 | ✓ (best single anchor) |
| Diversity floor (r vs strongest existing < 0.95) | yes | **0.944 vs BGE** | ✗ borderline — L7 redundancy zone |

The Pearson matrix told us before submitting:

| | e5 | bge | specter | scincl | ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5 | 1.000 | **0.944** | 0.916 | 0.909 | 0.832 |
| bge | 0.944 | 1.000 | 0.909 | 0.911 | 0.818 |

**Crucial insight: 5-anchor stacks (E5 + BGE + scincl + specter + ridge) underperformed 4-anchor stacks where E5 replaced BGE.** Best 5-anchor blend (e5 .35 / bge .25 / scincl .25 / specter .10 / ridge .05) had OOF round-QWK 0.6364. Best 4-anchor swap (e5 .40 / scincl .40 / specter .15 / ridge .05) had OOF round-QWK 0.6444. Adding E5 *next to* BGE wasted weight; replacing BGE with E5 captured the marginal signal gain.

This is the **same-family redundancy** pattern from L7 (3 SPECTER2 anchors at r=0.98 = one signal), but at the boundary (r=0.944). The rule still applies: **two encoders with r > ~0.93 should not both be in the stack**. Pick the stronger one and let it have the budget.

**Tuning timeline (every choice was forced by floor / probe rules):**

| Step | Anchor | OOF (single) | Test L1 (best blend) | Public LB | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 9 (BGE) | 0.6228 | 0.156 | 0.72374 | adopted (replace SciNCL primary) |
| 12 (E5) | 0.6417 | 0.142 | 0.72394 | adopted (replace BGE) |

**Sweep result (E5 4-anchor, ridge ≤ 0.10, 26 candidates):**

| Pick | weights (e5/scincl/specter/ridge) | OOF QWK | Test L1 | Public LB |
| --- | --- | ---: | ---: | ---: |
| BGE safest (prev best) | n/a | 0.6567 | 0.156 | 0.72374 |
| **E5 safest_e5** | 0.50 / 0.30 / 0.20 / 0.00 | **0.6593** | **0.142** | **0.72394** ⭐ |
| high_oof_e5 (not submitted) | 0.50 / 0.25 / 0.20 / 0.05 | 0.6615 | 0.149 | (predicted ~0.722-0.725) |
| swap_bge_e5 (not submitted) | 0.40 / 0.40 / 0.15 / 0.05 | 0.6577 | 0.162 | (sibling of BGE safest) |

The picked candidate has **ridge=0.00**. This is unusual — every prior public-best had ridge > 0 — but its test_L1 of 0.1422 was the lowest in any sweep we've ever run. Trusting L7/L9 (rank by test L1 first) was the right call: public lift was small but positive, not a regression.

**OOF→public ratio update:**

| Submission | OOF lift | Public lift | Ratio |
| --- | ---: | ---: | ---: |
| BGE 4-anchor safest (vs 0.72103 anchor) | +0.0155 | +0.0027 | 6× |
| **E5 4-anchor safest_e5 (vs BGE safest)** | +0.0026 | +0.0002 | **13×** |

The discount is widening as we approach the test ceiling. We are now in the regime where each +0.001 OOF buys ~+0.0001 public. **Anchor sweeps inside the contrastive-sentence-similarity family have effectively run out.**

**Rule of thumb (refines L7 + L13).**

- For two encoders with **r ∈ [0.93, 0.96]**, stacking both is wasteful. Replace the weaker (lower individual OOF) and let the stronger get the BGE/E5 budget.
- For r < 0.92, both can stack with a meaningful split (e.g., SciNCL r=0.911 vs E5 still got a 0.30 weight in safest_e5).
- **Diversity ceiling for the contrastive-sentence-similarity family is reached.** Any additional model in this family (`bge-m3`, `gte-large`, `nomic-embed`, `mxbai-embed-large`) will likely have r > 0.93 with E5 and not lift. This is now an experimental rule, not a prediction — confirmed by both BGE→E5 (small lift) and E5→???-future-encoder (predicted small).
- **Next anchor must come from outside the contrastive-sentence family** to break the ceiling. Options:
  - LLM fine-tune (Qwen/Llama LoRA) — generative regression, totally different geometry from CLS-mean encoders.
  - Cross-encoder fine-tune (e.g., `cross-encoder/ms-marco-MiniLM`) — ranks pair (paper, ASP-relevance prompt) jointly.
  - Pseudo-labeled retraining of E5 on confident test predictions (creates a different OOF distribution per L4 risk).

**Action items.**
- [x] Promote `next_e5_4anchor_safest_e5_submission.csv` (0.72394) as the new public anchor.
- [x] Document E5+BGE redundancy so the next experimenter doesn't try a 5-anchor stack again.
- [ ] **Do NOT fine-tune another contrastive-sentence-similarity encoder** (BGE-m3, GTE, Nomic, mxbai) — predicted r > 0.93 vs E5 = same-family ceiling. Would burn ~45 min GPU for ±0 public.
- [ ] Pivot to Qwen2.5-7B / Llama-3.1-8B LoRA (the original phase 2 plan) — generative regression has fundamentally different signal geometry; expected r vs encoders 0.6-0.8.
- [ ] If LoRA also caps, consider pseudo-labeling: take confident E5 test predictions (round_score < 1.2 or > 4.8) → label cardinality, retrain E5 on the 2,494 + ~150 augmented set. L4 risk applies.

---


## L14 — Public-validated weight rules are stack-configuration-specific, not universal

**Evidence (2026-05-18):** the `single_knob` follow-up to BGE 4-anchor `safest` (0.72374) regressed to **0.70965** on Public LB, a `−0.01409` drop.

| Submission | bge | scincl | specter | ridge | OOF | test_L1 | Public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `safest` (current best) | 0.40 | 0.40 | 0.15 | 0.05 | 0.6567 | 0.156 | **0.72374** |
| `single_knob` | 0.30 | 0.40 | 0.10 | 0.20 | 0.6572 | 0.158 | 0.70965 |
| Δ | −0.10 | 0 | −0.05 | **+0.15** | +0.0005 | +0.002 | **−0.0141** |

OOF and test_L1 both said "near-identical to safest". Public said otherwise. The dominant change vs safest is the **ridge weight 0.05 → 0.20**, a +0.15 jump that L10 had labelled as "public-safe" — but only in the *3-anchor* stack era.

**Why ridge=0.20 was safe for SciNCL/SPECTER/Ridge but not for BGE/SciNCL/SPECTER/Ridge:**
- In the 3-anchor stack, ridge was the *only* lexical signal; cutting it from 0.20 → 0.10 (the `0.71622` regression L10 documented) removed lexical coverage entirely.
- BGE-large brings BERT-large generic-text representations. Pearson r BGE vs Ridge = 0.818 — significantly more correlated than Ridge vs SPECTER2 (0.811). BGE partially **subsumes** ridge's role as the "lexical/surface-form" signal.
- With BGE present, ridge=0.20 becomes double-counting of lexical features, while ridge=0.05 lets BGE provide most of the lexical signal and ridge just calibrates edge cases.

This is structurally similar to the L7 finding (3 SPECTER2 anchors at r=0.98 are one signal). **Adding a new anchor with high correlation to an existing one redistributes which weights are safe.** The old rules don't carry over.

**Rule of thumb (refines L10 with a hard caveat).**
- L10's "single-knob sweeps near a public-validated point" assumed *the same anchor set*. When the anchor set changes (we added BGE for `safest`), the public-validated weight constraints from the previous configuration **do not transfer**.
- Treat the first public-confirmed weight tuple of a new stack-configuration as the **only** public-validated point, until further submissions test neighbours.
- For follow-up submissions, prefer single-knob neighbours of the **most recent public winner**, not of an earlier-era winner. After `safest` (40/40/15/05), valid single-knob neighbours are 35/40/15/10, 45/40/10/05, 40/45/10/05, etc. — *not* 30/40/10/20 (3 knobs moved at once).

**Action items.**
- [x] Mark `single_knob` as a regression in `leaderboard_tracking.md` so the next experimenter doesn't repeat the move.
- [ ] **Do NOT submit `high_signal` (25/25/25/25)** — its ridge weight is 0.25, even higher than the failed single_knob. Predicted public ~0.69-0.71. Save the daily slot.
- [ ] Re-sweep around `safest` with the constraint **ridge ≤ 0.10**. Prefer test_L1 ≤ 0.16. New candidates worth probing:
  - 40/40/10/10 (single-knob: ridge ↑0.05, specter ↓0.05)
  - 40/45/10/05 (single-knob: scincl ↑0.05, specter ↓0.05)
  - 45/40/10/05 (single-knob: bge ↑0.05, specter ↓0.05)
  - 35/45/15/05 (single-knob: bge ↓0.05, scincl ↑0.05)
  - 40/40/20/00 (test ridge=0 — see if ridge contributes anything when bge is high)
  - 40/30/25/05 (already in candidates.csv at OOF 0.6576, test_L1 0.162 — borderline test_L1)
- [ ] After confirming/refuting one of these, only then expand to e5-large-v2 / Qwen LoRA.

---


## L13 — BGE-large-en-v1.5 cleared the L11/L12 floors and lifted the stack; OOF→public discount was 6×

**Evidence (2026-05-18):** fine-tuned `BAAI/bge-large-en-v1.5` (335M, BERT-large arch, mean pool, bf16) on `title + abstract`, 5 folds × 3 seeds = 15 models. First text encoder since SciNCL (May 14) to lift the stack.

**Floor checks (per L12):**

| Floor | Required | BGE | Pass? |
| --- | --- | --- | --- |
| Signal floor (OOF round-QWK ≥ 0.51) | yes | 0.6228 | ✓ |
| Diversity floor (r vs SPECTER2 < 0.95) | yes | 0.909 | ✓ |

The diversity number is the key. SciBERT (0.948), SPECTER2 v2 (0.98+), and SciNCL (0.943) all sat above 0.94; BGE at 0.909 is the first text encoder to live in the 0.85-0.92 sweet spot **and** clear the signal floor at the same time.

**Why BGE worked where DeBERTa-v3 didn't (a model lineage lesson):**
- DeBERTa-v3 was pre-trained with **Replaced Token Detection** (ELECTRA-style) on CommonCrawl + Wikipedia + Books. No semantic-similarity inductive bias. Generic large.
- BGE was pre-trained with **RetroMAE + contrastive sentence-pair similarity** on a massive web-pair corpus. The objective is structurally close to SPECTER2's triplet citation contrastive — both push semantically related sequences together in embedding space.
- The pre-training objective predicted transfer better than capacity. BGE (335M) lifted; DeBERTa-v3 (435M) didn't. **Objective family > parameter count.**

**Stacking sweep (32 weight combinations) and pick:**

| Candidate | OOF QWK | Test L1 | Public LB |
| --- | ---: | ---: | ---: |
| 60/20/20 anchor (no bge) | 0.6412 | 0.162 | 0.72103 |
| **safest: bge 0.40 / scincl 0.40 / specter 0.15 / ridge 0.05** | **0.6567** | **0.156** | **0.72374** |
| single_knob: 0.30 / 0.40 / 0.10 / 0.20 | 0.6572 | 0.158 | not submitted |
| high_signal: 0.25 / 0.25 / 0.25 / 0.25 | 0.6593 | 0.172 | not submitted |

L7/L9 ranking (test L1 first) picked `safest`. **Lift was real but smaller than predicted:** OOF +0.0155 → public +0.0027, an OOF→public discount of ~6×.

**Why the discount was so large:**
- SciNCL anchor (`0.72103`, May 16) had OOF *negative* (−0.0050) but public *positive* (+0.011). Different sign pattern.
- BGE anchor (`0.72374`, this) had OOF positive (+0.0155) and public positive but small (+0.0027).
- Public split is ~50% of test (596 of 596 rows visible to LB), private is the other 50%. We changed the **distribution** more than the SciNCL pick did (combined dist `{1:256, 2:129, 3:101, 4:66, 5:44}` vs anchor `{1:264, 2:123, 3:99, 4:67, 5:43}`, swapping ~10 rows around label 1↔2 and 3↔4). Public was relatively insensitive to those particular swaps.

**Rule of thumb (refines L7).**

- The OOF→public ratio is **not** stable across submissions. We've seen 4 distinct patterns now in `outputs/leaderboard_tracking.md`:
  - Stack-improvement-by-distribution (label4_more_mid: OOF +0.001, public +0.017): predicted distribution change is the dominant lever.
  - Stack-improvement-by-anchor-replacement (SciNCL `0.72103`: OOF −0.005, public +0.011): a new diverse signal can flip the OOF→public sign.
  - Stack-improvement-by-anchor-addition (BGE `0.72374`: OOF +0.015, public +0.003): a new anchor with high diversity-clear lifts both, but with shrinkage.
  - Anchor-redundancy regression (3-seed → 5-seed SPECTER2: OOF +0.006, public −0.012): more of the same signal hurts.
- Submission picks should still rank by test L1 first, but **expectations on the lift should be set conservatively** — assume a 3-6× OOF→public discount unless predicted distribution moves substantially in the same direction as a *previously-confirmed-good* shift (e.g., the label-4 expansion that gave +0.017).
- One submission per day is the right pace. Don't fire 3 candidates at once: each result narrows the next pick, and submission slots are scarce.

**Action items.**
- [x] Add BGE-large to the standard anchor toolkit. Recipe (`notebooks/step9_bge_large_finetune.ipynb`) is hardened from step 8 lessons (fp32 load, mean pool, NaN guard).
- [ ] Submit `single_knob` (30/40/10/20) next — has near-identical OOF/L1 to `safest` but keeps ridge at 0.20 (L10 public-safe). Either it confirms the winning region or it tells us the bge=0.40 specifically matters.
- [ ] Try `e5-large-v2` as a 5th anchor (same family as BGE — contrastive sentence similarity).
- [ ] When we have 5 anchors, do a single round of meta-Ridge stacking *with strong L1 regularization* on the 5 OOFs. Don't repeat the L7 mistake (top-OOF meta-model dropped the public-validated anchor) — keep BGE/SciNCL weights bounded ≥ 0.20 each.

---


## L12 — Non-text anchors also fail the signal-floor rule; diversity is necessary but not sufficient

**Evidence (2026-05-18):** built an OpenAlex continuous OOF anchor (Huber regressor over DOI + title-search citation features + venue/year/author target encoding, 5 folds × 3 seeds = 15 models). Despite being the **most diverse** candidate ever tried, it failed the blend probe.

| Anchor | OOF round-QWK | Pearson r vs SPECTER2 | Pearson r vs SciNCL |
| --- | ---: | ---: | ---: |
| SPECTER2 | 0.6297 | 1.000 | 0.943 |
| SciNCL | 0.6117 | 0.943 | 1.000 |
| Ridge 5×5 | 0.4353 | 0.811 | 0.817 |
| **OpenAlex anchor** | **0.2570** | **0.499** | **0.505** |

`r ≈ 0.5` is by far the lowest correlation in the project (vs DeBERTa-v3 0.848, SciBERT 0.948, LLM v2 0.558). **Diversity alone is not enough:**

| Mix | round-QWK | Δ vs anchor 0.6074 |
| --- | ---: | ---: |
| 60/20/20 anchor (no openalex) | **0.6074** | (baseline) |
| `ridge −0.05, +openalex 0.05` | 0.6082 | **+0.0008** ← noise band |
| ridge −0.10, +openalex 0.10 | 0.6033 | −0.004 |
| scincl/specter −0.025, +openalex 0.05 | 0.5964 | −0.011 |
| any weight ≥ 0.10 from any anchor | 0.55-0.60 | clear regression |

The single +0.0008 cell is below seed-to-seed variance. No usable lift exists.

**Cross-cutting pattern (now confirmed across 4 distinct candidate types):**

| Candidate | Type | OOF round-QWK | r vs SPECTER2 | Verdict | Lesson |
| --- | --- | ---: | ---: | --- | --- |
| LLM v2 zero-shot | text, generic LLM | 0.3752 | 0.558 | drop | L8 |
| SciBERT fine-tune | text, scientific BERT | 0.6165 | **0.948** | drop (redundant) | L7 |
| DeBERTa-v3-large fine-tune | text, generic large | 0.5511 | 0.848 | drop | L11 |
| **OpenAlex anchor** | **non-text, citation/venue** | **0.2570** | **0.499** | **drop** | **L12** |

Every one fails the blend probe. The reason is the same in all four: **signal_strength × diversity** must clear *both* thresholds, not just one.

**Refined rule of thumb (supersedes the loose version in L8).**

For a candidate to lift the stack, it must satisfy *both*:
1. **Signal floor:** OOF round-QWK ≥ ~80% of the strongest current anchor's OOF (currently ~0.51 on this dataset). Below that, the candidate carries more noise than signal.
2. **Diversity floor:** Pearson r vs the strongest anchor < ~0.95 (above that, the candidate is essentially the same signal — see L7).

OpenAlex passed (2) by a wide margin (r 0.50 << 0.95) but failed (1) catastrophically (0.26 vs 0.51). DeBERTa-v3 / SciBERT / LLM each failed at least one floor. The anchors that *did* lift the stack — Ridge 5×5 (0.4353 OOF, r 0.811) and SciNCL (0.6117 OOF, r 0.943) — barely cleared one floor but had a clear margin on the other.

**Why the OpenAlex signal is so weak on this dataset.** The label is "ASP/AI-symbolic relevance of an abstract" (L5), not "paper quality". Citation count and FWCI track impact, not topic relevance. A heavily-cited proceedings volume is label 1; a niche label-5 ASP solver paper has near-zero citations. This is the same trap that L3 flagged for the all-source scholarly dump.

**Action items.**
- [x] Drop OpenAlex anchor for stacking. Keep `outputs/openalex_anchor/` as a negative reference and a known-low-correlation source for future combo experiments.
- [ ] Codify the **+10% blend probe** as a mandatory preflight before committing to any new anchor's full 15-fold training run. Run it on whatever cheap proxy OOF is available first (e.g., a 1-seed × 5-fold version), and abort if the round-QWK does not match or beat the current 60/20/20 anchor at *any* probed weight.
- [ ] When experimenting with non-text features, do **not** treat them as a separate anchor. Instead, blend them *into* the Ridge anchor's feature set (the way the original 0.63064 OpenAlex Huber meta did with `ridge_score` as a feature). The meta-blend approach worked publicly; the standalone anchor approach does not.
- [ ] Next encoder candidate worth trying: **`bge-large-en-v1.5`** or **`e5-large-v2`** — generic large encoders pre-trained with **contrastive sentence similarity** (closer to SPECTER2's objective than DeBERTa-v3's RTD). Predicted: clear signal floor (≥ 0.55 OOF), correlation 0.85-0.92.

---


## L11 — Generic large encoders (DeBERTa-v3) under-fit scientific-paper datasets, regardless of tuning

**Evidence (2026-05-18):** fine-tuned `microsoft/deberta-v3-large` (435M params) on the same `title + abstract` input, 5 folds × 3 seeds. After applying every standard DeBERTa-v3 fine-tune trick the literature recommends, OOF QWK plateaued at **0.5511** (round) / **0.5689** (constrained-tuned) — well below SPECTER2's 0.6297 and SciNCL's 0.6117 on the same OOF rows.

**Tuning timeline (every fix moved the needle, none rescued the gap):**

| Variant | best per-fold |
| --- | ---: |
| CLS pool, fp16/bf16 autocast | NaN loss (disentangled attention overflow) |
| CLS pool, fp32, LR_enc 1e-5, 5 epochs | 0.502 |
| CLS pool, fp32, LR_enc 2e-5, 6 epochs | 0.533 |
| Mean pool, fp32, LR_enc 2e-5, 6 epochs | 0.570 |
| Mean pool + LLRD 0.95, 5 epochs | **0.580** |

Engineering notes (will save the next person hours):
- DeBERTa-v3's `model.safetensors` on the Hub is **fp16**. `transformers >= 4.41` honours that, so `AutoModel.from_pretrained` returns an fp16 encoder; a freshly-built fp32 head then crashes with `mat1 and mat2 must have the same dtype`. Pass `torch_dtype=torch.float32` (or `dtype=` on transformers v5).
- Disentangled attention overflows in fp16 *and* bf16. Use **pure fp32** for both train and eval. The 1.4× slowdown is unavoidable.
- DeBERTa-v3 was pre-trained with **Replaced Token Detection** (ELECTRA-style), not NSP/contrastive on `[CLS]`. The position-0 token is **not** a sequence summary, so CLS pooling under-performs by ~0.04 QWK vs masked **mean pooling** over all tokens. This is the FB3/ELL community standard.
- Layer-Wise LR Decay (LLRD ≈ 0.95) on top of mean pooling adds another ~0.01 — the canonical last-mile fix.
- Head LR `1e-3` (the BERT-base default from step 6) blows up deberta-large to NaN. Use `1e-4`.

**Stacking probe (definitive evidence, not a heuristic):**

| Anchor | OOF round-QWK | Pearson r vs SPECTER2 | Pearson r vs SciNCL |
| --- | ---: | ---: | ---: |
| SPECTER2 | 0.6297 | 1.000 | 0.943 |
| SciNCL | 0.6117 | 0.943 | 1.000 |
| Ridge 5×5 | 0.4353 | 0.811 | 0.817 |
| **DeBERTa-v3** | **0.5511** | **0.848** | **0.859** |

Round-QWK of `0.6 * scincl + 0.2 * specter + 0.2 * ridge` (the public-best anchor) **regresses monotonically** when DeBERTa-v3 is added at any weight, before any threshold tuning:

| Mix | OOF round-QWK |
| --- | ---: |
| 60/20/20 anchor (no deberta) | **0.6074** |
| + deberta 5% (taken evenly from scincl/specter) | 0.6037 |
| + deberta 10% | 0.5997 |
| + deberta 15% | 0.5983 |
| + deberta 20% | 0.5978 |

**Why this matches L8 (LLM zero-shot drop) almost exactly:**

| Candidate | OOF | Pearson r vs SPECTER2 | Verdict |
| --- | ---: | ---: | --- |
| LLM v2 zero-shot (L8) | 0.3752 | 0.558 | dropped |
| **DeBERTa-v3 large (this)** | **0.5511** | **0.848** | **dropped** |

DeBERTa-v3 has stronger raw signal than the LLM but worse diversity. The product (signal × diversity) is in the same dead zone — net effect on the stack is negative.

**Root cause.** SPECTER2 / SciNCL / SciBERT were pre-trained on scientific paper similarity (citation contrastive objectives, scientific abstract corpora). DeBERTa-v3 was pre-trained on CommonCrawl + Wikipedia + Books — **generic English**, no scientific-paper inductive bias. On a 2,494-row supervised target where the label axis is "ASP/AI-symbolic relevance of an abstract", the domain-specific pretraining matters far more than encoder capacity (435M vs 110M).

**Rule of thumb (revises L7, L8, L9 for the encoder choice).**

- For scientific-paper datasets, **prefer encoders pre-trained on scientific corpora** (SPECTER2, SciNCL, SciBERT, allenai/specter*, OAG-BERT). Generic large encoders (DeBERTa-v3, RoBERTa-large, ELECTRA-large) start ~0.05-0.08 OOF QWK behind and tuning does not close the gap.
- Diversity benefit kicks in only above an OOF floor of **~80% of the strongest anchor's OOF** (already L8). Below that floor, the candidate is dead weight regardless of correlation.
- Before training a new candidate end-to-end, **run a quick blend probe with its OOF**: if the no-threshold round-QWK of `(anchor mix + 0.10 * candidate)` is below the anchor's round-QWK, do not finish the run — abort and try a different signal source.

**Action items.**
- [x] Drop step 8 (DeBERTa-v3-large fine-tune). Keep OOF on disk as a negative reference.
- [ ] If a 4th anchor is still wanted, do **not** try `roberta-large` / `electra-large` — same architecture failure mode predicted. Try instead:
  - `bge-large-en-v1.5` or `e5-large-v2` (large generic encoders, but pre-trained with **contrastive sentence similarity** — closer to SPECTER2's objective, so signal may transfer).
  - Non-text signals: OpenAlex citation OOF as a continuous anchor, venue / first-author target encoding.
- [ ] Add a "quick blend probe" step to every future anchor experiment: compute the `+10%` OOF mix before committing to the full 15-fold run.

---


## L10 — When sweeping weights around a public-validated anchor, change ONE knob at a time

**Evidence (2026-05-18):** swept 17 weight combinations around the public-best 60/20/20 (`scincl 0.6 / specter 0.2 / ridge 0.2`, public 0.72103). Picked `70/20/10` for submission because it had the lowest test L1 distance (0.1584 vs 60/20/20's 0.162). It scored Public LB `0.71622`, a `−0.00481` regression.

| Variant | Weights | OOF QWK | Test L1 | Public LB |
| --- | --- | ---: | ---: | ---: |
| 60/20/20 (anchor) | scincl 0.6 / specter 0.2 / ridge 0.2 | **0.6412** | 0.162 | **0.72103** |
| 70/20/10 | scincl 0.7 / specter 0.2 / ridge 0.1 | 0.6393 | 0.158 | 0.71622 |

The 70/20/10 candidate moved **two knobs at once** vs the anchor: SciNCL up by `0.10` and Ridge down by `0.10`. The Ridge cut from 20% to 10% appears to have hurt more than the SciNCL boost helped, even though OOF and test L1 both said "this should work".

**Rule of thumb (refines L6 + L9):**

- For sweeps near a public-validated point, prefer single-knob moves: `65/20/15`, `60/15/25`, `60/25/15`, `55/20/25`, etc., where exactly one weight changes vs the anchor.
- A multi-knob move that improves both heuristics (lower test L1 *and* equal-ish OOF) can still regress; the heuristic-vs-public correlation is weaker than the single-anchor case.
- When in doubt, prefer keeping the public-best anchor unless a candidate has *both* a clear OOF win (>0.005) AND a test L1 inside the same band (within 0.01).

**Action items.**
- [x] Future submission picks should hold ridge weight constant at 0.20 (proven public-safe) when sweeping the BERT-class weights.
- [ ] Next worth-trying single-knob variants: `65/20/15`, `55/20/25`, `60/15/25`, `60/25/15`.

---


## L9 — Test L1 distance and OOF QWK are heuristics, not laws; blending strong correlated anchors can still help

**Evidence (2026-05-16):** `0.72103` (`blend_3anchor_scincl_specter_ridge_60_20_20`) beat `0.71052` (`blend_2anchor_70_specter`) on public LB by `+0.0105` despite **both metrics suggesting it should regress**:

| Metric | 0.71052 anchor | 0.72103 (this submission) |
| --- | ---: | ---: |
| OOF QWK (constrained tuner) | **0.6464** | 0.6412 (**lower**) |
| Test combined L1 distance vs train | **0.152** | 0.162 (**higher**) |
| Predicted label dist | similar | similar |

L7 said "rank by test L1 first, OOF second"; both metrics here pointed to "do not submit". Yet public went up.

**What this means.**
- Test L1 distance is a useful *filter* but not a deterministic predictor. It works to flag drifted distributions (5-seed run 0.68737 had test L1 0.183, public regressed); it does not reliably rank candidates that are all in the "well-calibrated" zone (test L1 0.13 - 0.18).
- OOF QWK on a single CV fold is also noisy. A 0.005 OOF gap is comparable to the seed-to-seed variance in fine-tune training, so it should not by itself disqualify a candidate.
- A new anchor with OOF below the existing best can still lift the stack if it adds enough independent signal. **High pairwise correlation does not mean redundancy.** SciNCL had Pearson `r = 0.943` with SPECTER2 (similar to SciBERT's 0.948) and OOF `0.6269` < SPECTER2's `0.6373`, yet the 60/20/20 blend out-performs SPECTER2 + Ridge.

**Why SciNCL worked while SciBERT did not (despite similar correlation).**
- SciBERT and SPECTER2 share the same MLM checkpoint; SPECTER2 is SciBERT plus a contrastive paper-similarity head. Their disagreement set is small.
- SciNCL was trained from scratch with a *different* contrastive objective (neighborhood contrastive learning over the citation graph). Its disagreement with SPECTER2 is on different rows than where SciBERT disagrees, even though the correlation is similar.
- We do not have an explicit metric for "kind of disagreement". The lesson is: do not rule out a candidate purely on correlation.

**Rule of thumb (revised L7).**
- Treat OOF QWK gap < 0.01 and test L1 distance gap < 0.03 as *noise* relative to the current anchor.
- For two anchors with `r > 0.9`, still try a blend if both are individually competitive (OOF >= 80% of the strongest anchor's OOF).
- Submit borderline candidates **one at a time** to learn which signal pattern is genuine. Burning a daily slot on a "lowest test L1 within the new family" candidate is a reasonable hedge even when both metrics formally favour the anchor.

**Action items.**
- [ ] If a new base model is added, evaluate a 60/20/20 weighting scheme (or a small 50-65 / 15-25 / 15-25 grid) before discarding it for redundancy.
- [ ] Keep test L1 distance as a *flag*, not a *gate*. Reject only candidates with test L1 above some absolute threshold (~ 0.18 looks safe; > 0.20 has consistently hurt).
- [ ] Sweep the 3-anchor weights around 60/20/20 (55/25/20, 65/15/20, 70/10/20) on the same anchors to confirm the winning region.

---


## L8 — LLM zero-shot has the diversity but not the signal strength

**Evidence (2026-05-16):** ran DeepSeek `deepseek-v4-flash` zero-shot scoring (verbalizer trick over digit token logprobs) on the full 3090 papers. Two prompt versions:

| | LLM v1 (5 examples, no reasoning) | LLM v2 (7 examples + reasoning lines) |
| --- | ---: | ---: |
| Mean score for true=1 | 1.62 | 1.59 |
| Mean score for true=5 | 3.47 | 3.27 |
| Spread (5 - 1) | 1.85 | **1.68** (worse) |
| Constrained-tuned OOF QWK | 0.3704 | 0.3752 |
| Pearson r vs SPECTER2 | **0.569** | **0.558** |

**Key finding:** the LLM has **excellent diversity** (correlation 0.55-0.57 vs SPECTER2, much lower than the 0.948 we saw between SPECTER2 and SciBERT) but **insufficient signal strength** (OOF 0.37 vs SPECTER2 0.64).

In stacking, the best LLM-blended candidates lose to the current `blend_2anchor_70_specter` best:

| Candidate | OOF QWK | Test L1 |
| --- | ---: | ---: |
| blend_2anchor_70_specter (current best, public 0.71052) | **0.6464** | **0.152** |
| blend_3anchor_srl_60_25_15 (60% specter + 25% ridge + 15% LLM v2) | 0.6364 | 0.213 |
| blend_specter_llm_70_30 (70% specter + 30% LLM v2) | 0.6117 | 0.240 |
| anchor_llm_zs_only (LLM v2 alone) | 0.3752 | 0.598 |

Adding the LLM column to any blend pulls test L1 distance up by ~0.05-0.08 because LLM scores are biased toward labels 1-2 (the model "plays safe" on a niche domain it does not know well).

**Why prompt v2 was no better than v1.** Adding more label-5 examples + explicit reasoning lines made the model *more* conservative on label 5, not less. With 7 examples, the few-shot pattern reinforced "look for explicit ASP+neural cues" so the model only emits 5 when the title literally says ASP and neural in the same line. Most label-5 papers in this dataset are about ASP solver / semantics work where the title does not mention "neural", and the LLM under-rates those.

**Root cause.** Zero-shot LLM scoring of a niche academic relevance scale, where ground-truth labels were assigned by a human annotator with task-specific judgement, cannot beat a fine-tuned encoder that saw the labels during training. SPECTER2 fine-tune > SciBERT fine-tune > Ridge TF-IDF >> LLM zero-shot, in that order, on this dataset.

**Rule of thumb.**
- For supervised-style anchors in stacking, the new anchor needs OOF QWK of at least about **80% of the strongest existing anchor's OOF**, even when its correlation is low. Below that threshold, the noise penalty dominates the diversity benefit.
  - SPECTER2 OOF 0.64 -> threshold ~ 0.51
  - LLM v1 0.37, v2 0.38 -> well below threshold; predicted to hurt stack; confirmed.
- Better diversity does **not** rescue weak signal. Diversity matters only when the anchors are individually competitive.
- For LLM zero-shot to be useful here, we would need a model that can match SPECTER2's domain-specific calibration. That probably requires fine-tuning the LLM (defeats zero-shot) or chain-of-thought prompts that DeepSeek-flash truncates with thinking-mode token budget.

**Action items.**
- [x] Stop adding text-only base anchors. Both SciBERT (0.948 corr, +0 stack gain) and LLM zero-shot (0.55 corr, but OOF 0.37) confirm we are at the text-feature ceiling on this dataset.
- [ ] Pivot to **non-text** signals: re-extract OpenAlex citation OOF as a continuous anchor; add venue/year-prior anchor; add first-author target encoding anchor.
- [ ] Or accept 0.71052 as the final answer and write up the report.

---


submissions and the rule of thumb to avoid repeating the mistake.

## L7 — Stack by *test L1 distance*, not OOF QWK; minimalist 2-anchor beats 4-anchor

**Evidence (2026-05-15):** the stacking step (`outputs/stacking/`) tried 12 candidates over 4 anchors (ridge_5x5 + 3 SPECTER2 finetunes). Two key findings:

1. The 3 SPECTER2 anchors had pairwise Pearson correlation 0.98+. They are essentially the same signal. Only ridge_5x5 (correlation 0.81 vs SPECTER2) is a genuinely different second signal.

2. Meta-models that select weights by OOF (huber, ridge, constrained blend) consistently picked `specter_5s` and `specter_v2` over `specter_3s` because they had ~0.005 higher OOF QWK. These same meta-models gave `specter_3s` weight 0 or negative. Yet `specter_3s` was the public-best anchor (0.69972 vs 0.687). They optimised for OOF and dropped the public-validated signal.

3. **Test combined L1 distance** (predicted label distribution vs train distribution) is a much better proxy for public LB than OOF QWK on this dataset:
   - Top-OOF candidate `huber_meta` had OOF 0.6534 but test L1 = 0.179. Likely public regression.
   - Selected candidate `blend_2anchor_70_specter` had OOF 0.6464 (lower) but test L1 = 0.152 (lowest). **Public LB 0.71052** (+0.011 over previous best).

**Root cause.** When base anchors are highly correlated, OOF rewards picking the slightly-better-OOF version, but those marginally different versions transfer marginally differently to test. The variance dominates the OOF gain.

**Rule of thumb.**
- For each pair of anchors, compute Pearson correlation on OOF. Treat anchors with r > 0.9 as the same signal.
- A minimalist 2-anchor stack of *truly different* signals (correlation 0.6-0.85) is usually safer and stronger than a 4+ anchor stack with redundant base models.
- When picking the candidate to submit, **rank by test combined L1 distance first**, OOF QWK second. Both must be acceptable: OOF QWK >= base anchor, test L1 close to or below the anchor's test L1.
- Bias the blend weight toward the public-validated anchor (e.g. 0.70 weight for the anchor with the strongest public LB), not toward the highest-OOF anchor.

**Action items.**
- [ ] Future stacks should add a *truly diverse* anchor (SciBERT fine-tune, LLM scoring) rather than 5-seed / 7-seed re-runs of SPECTER2.
- [ ] Add a column "Pearson r vs strongest anchor" to any anchor candidate before adding to a stack.
- [ ] Always include `anchor_specter_3s_only` (or whatever the current public-best is) as a sanity-baseline candidate inside the stacking script.

---



**Evidence (2026-05-15):** `0.68718` (3c run). Three changes were applied at the same time vs the 0.69972 baseline:

1. v2 abstract cache (93.7%) -> v3 cache (95.7%, +44 abstracts).
2. `MAX_LEN` 256 -> 384.
3. Unconstrained threshold tuner -> constrained (lambda=0.5 * L1 distance).

OOF QWK rose to 0.6446 (+0.006). OOF L1 distance to train dist was excellent (0.0225). Yet public LB dropped 0.69972 -> 0.68718 (-0.01254).

42 test rows changed: 29 down, 13 up. Public split absorbed 29 changes; private only 13. Net direction: the model became more conservative.

**Root cause.** Each knob individually looked safe, but they interacted. The constrained tuner protected OOF distribution; v3 cache + max_len 384 changed the *test* score distribution; and we cannot tell which of the three was the actual culprit because we changed them together.

**Rule of thumb.**
- Change one knob at a time. After each change, submit and confirm public direction *before* the next change.
- The constrained tuner is the one knob that has independent OOF evidence (L1 from 0.1275 -> 0.0225) — keep it.
- For abstract cache + max_len, run separate ablations next time. Hypothesis: max_len 256 with v3 cache is probably better than max_len 384 with v3 cache, because most label-bearing signal is in the first 200 tokens of the abstract anyway.

**Action items.**
- [ ] Next experiment: change *only* the abstract cache (v2 -> v3), keep everything else identical to 0.69972. If OOF improves and public improves -> v3 is good. Otherwise revert.
- [ ] Separately: keep v2 cache + change *only* max_len 256 -> 384. Compare.
- [ ] Do NOT submit a multi-change run unless each underlying change has been individually validated.

---

## L1 (updated) — Constrained tuner is necessary but not sufficient

The constrained tuner *did* fix OOF distribution (L1 dist 0.0225 in 0.68718 vs 0.139 in unconstrained 5-seed). But it did **not** prevent public regression, because the encoder change shifted the test score distribution upstream of threshold tuning.

So L1's rule "predicted distribution should stay close to train" applies to **test** distribution, not OOF. Achieving low OOF L1 is necessary but not sufficient. Future Action: also verify *test* combined distribution L1 < ~0.12 before submitting.

---

## L1 — OOF QWK can rise while public LB falls (threshold-distribution drift)

**Evidence (latest, 2026-05-13):** `0.69972` (3-seed SPECTER2 fine-tune) vs
`0.68737` (5-seed rerun, same notebook, only `SEEDS` differs).

| Run | OOF QWK | Public LB |
| --- | ---: | ---: |
| 3 seeds | 0.6389 | **0.69972** |
| 5 seeds | **0.6445** | 0.68737 |

OOF +0.0056, public **−0.01235**. The 5-seed run beat itself on OOF and lost
on public.

**Earlier evidence (same pattern, May-12):** 5x5 Ridge threshold (`0.60804`,
OOF 0.5967) vs 10x10 Ridge threshold (`0.59612`, OOF 0.6051). 10x10 had higher
OOF but lower public.

**Root cause.** `scipy.optimize.differential_evolution` is greedy on OOF QWK
with only a `min_gap >= 0.03` constraint. When the score distribution shifts
slightly (more seeds = tighter distribution), the optimiser finds thresholds
that improve OOF by reshuffling 50-100 borderline rows between adjacent
labels, but those reshuffled labels do not match the public split's true
label proportions. The predicted label distribution **drifts** away from the
training distribution.

**Quantitative test.**

- Train distribution: `{1: 36%, 2: 21%, 3: 18%, 4: 15%, 5: 11%}`.
- 0.69972 public predicted: `{1: 39%, 2: 22%, 3: 20%, 4: 11%, 5: 9%}` —
  within 4 percentage points of train.
- 0.68737 public predicted: `{1: 29%, 2: 36%, 3: 16%, 4: 10%, 5: 9%}` —
  label 2 inflated to ~2x its expected share.

**Rule of thumb.**

- Treat public LB as ground truth, not OOF. When OOF improves but public
  drops, *do not* promote the new model.
- When tuning thresholds, the predicted label distribution must stay close to
  the training distribution. A useful soft rule: `|pred_share - train_share|
  <= 0.05` per label.
- Differential evolution alone is not safe. Add a **distribution-distance
  penalty** to the objective: `-QWK + lambda * sum_k |pred_share_k -
  train_share_k|` with `lambda` around `0.5`. Or restrict the per-threshold
  search bounds tighter once a public-validated solution is known.
- Constrained submissions are private-safe even if OOF is slightly worse.

**Action items.**

- [ ] Step 6: implement distribution-constrained threshold tuning that
      reproduces the `0.69972` threshold neighbourhood and rejects shifts
      that exceed the soft 5-percentage-point rule.
- [ ] Snapshot the 0.69972 thresholds `[1.808, 2.457, 3.230, 4.132]` as the
      anchor for any future score-on-the-same-distribution submission.

---

## L2 — Frozen pretrained encoders collapse minor-class predictions

**Evidence:** Step 3a (frozen SPECTER2 + Ridge head) reached OOF QWK only
`0.49`, with label-5 collapsed to **14 / 596** rows of test predictions.
Step 3b (fine-tuned SPECTER2 + regression head) reached OOF QWK `0.6389`
with label-5 at **51 / 596** rows — matching the train distribution.

**Root cause.** Pretrained `[CLS]` embeddings are optimised for similarity,
not for direct ordinal regression on a domain-specific axis like
"ASP-relevance". Linear/Ridge heads cannot recover that axis cleanly from a
generic 768-dim space, so high-confidence label-5 predictions collapse.

**Rule of thumb.** Whenever a pre-trained encoder is the centre of a model,
fine-tune the encoder end-to-end on the labelled training set. Do not rely on
frozen embeddings + linear head for tasks where the label axis is not the
dominant axis of the pretraining objective.

---

## L3 — Aggressive metadata blending is brittle

**Evidence (May-12):**

| Submission | Public LB | OOF QWK | Note |
| --- | ---: | ---: | --- |
| OpenAlex DOI Huber meta | **0.63064** | 0.6067 | only DOI-only |
| All-source scholarly meta | 0.60531 | 0.6025 | S2 + Crossref + COCI dump |
| Filtered COCI meta | 0.61227 | **0.6081** | high OOF, public unstable |

The high-OOF filtered candidate (`0.6081`) lost to the simpler DOI-only model
(`0.6067`) on public LB by `0.018`. Adding more external sources actively
hurt.

**Rule of thumb.** External features are noisy. Add them one at a time, only
when they correlate with the label after controlling for the existing anchor.
Do not push aggregate "all-source" feature dumps to leaderboard.

---

## L4 — Higher CV folds / seeds are not free

**Evidence:** 5x5 Ridge (`0.60804`) vs 10x10 Ridge (`0.59612`). More
folds/seeds had higher OOF (`0.6051` vs `0.5967`) but worse public.

**Rule of thumb.** Use enough folds/seeds to stabilise the OOF estimator, then
stop. Beyond a point, extra folds/seeds shift the score distribution, threshold
tuning re-optimises against the new distribution, and the predicted labels
drift. The 5x3 SPECTER2 fine-tune (15 models) is a sweet spot; 5x5 (25 models)
already drifts.

## L6 — Changing multiple knobs at once = un-attributable regression

**Evidence (2026-05-15):** `0.68718` (3c run). Three changes were applied at the same time vs the 0.69972 baseline:

1. v2 abstract cache (93.7%) -> v3 cache (95.7%, +44 abstracts).
2. `MAX_LEN` 256 -> 384.
3. Unconstrained threshold tuner -> constrained (lambda=0.5 * L1 distance).

OOF QWK rose to 0.6446 (+0.006). OOF L1 distance to train dist was excellent (0.0225). Yet public LB dropped 0.69972 -> 0.68718 (-0.01254).

42 test rows changed: 29 down, 13 up. Public split absorbed 29 changes; private only 13. Net direction: the model became more conservative.

**Root cause.** Each knob individually looked safe, but they interacted. The constrained tuner protected OOF distribution; v3 cache + max_len 384 changed the *test* score distribution; and we cannot tell which of the three was the actual culprit because we changed them together.

**Rule of thumb.**
- Change one knob at a time. After each change, submit and confirm public direction *before* the next change.
- The constrained tuner is the one knob that has independent OOF evidence (L1 from 0.1275 -> 0.0225) — keep it.
- For abstract cache + max_len, run separate ablations next time.

**Action items.**
- [ ] Future experiments must isolate one variable at a time.
- [ ] Do NOT submit a multi-change run unless each underlying change has been individually validated.

---

---

## L5 — Label = ASP / AI-symbolic relevance, not paper quality

**Evidence:** Step 1 keyword-vs-label table:
- `asp_core` keywords: train mean Label `3.92` vs `2.29` (delta **+1.63**).
- `ai_symbolic`: delta +1.23.
- `low_signal_title` (proceedings, short paper, invited): delta **−1.08**.
- `logic_neighbour` (Prolog/Datalog/Argumentation/DescLogic without ASP):
  delta only +0.32.

Step 3b reverse-analysis (`outputs/eda_v2/label_meaning_from_specter2.md`)
confirmed:
- Label 1 = proceedings / off-topic.
- Label 2 = adjacent (verification, KR) without ASP.
- Label 3 = generic logic / declarative reasoning.
- Label 4 = applied / extending ASP.
- Label 5 = core ASP advances + ASP × AI cross-overs (neuro-symbolic, NN
  verification, explainable AI built on ASP).

**Rule of thumb.** Any future feature engineering should target this axis:
- ASP-specific terms (`asp`, `answer set`, `clingo`, `dlv`, `stable model`,
  `aspq`, `grounder`).
- AI-symbolic combiners (`neuro-symbolic`, `explainable`, `neural` +
  `verification`).
- Anti-features for label 1 (`proceedings`, `workshop summary`, `invited
  talk`, `(short paper)`, `(extended abstract)`).

---

## Quick checklist before submitting any new model

1. OOF QWK improves over the current anchor by **>0.005**.
2. Predicted label distribution is within **5 percentage points** of train per
   label.
3. Public/private predicted label-5 split is balanced (e.g. 26/25), not
   skewed (e.g. 40/10).
4. The model uses a different signal source from the current anchor, so the
   submission contributes diversity for stacking even if the public score is
   close.
5. If the OOF improves but the predicted distribution drifts, **do not
   submit**. Retune thresholds with a distribution penalty first.
