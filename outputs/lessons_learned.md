# Lessons learned — for the next agent / future-me

Living document. Each entry records a concrete observation we paid for in
submissions and the rule of thumb to avoid repeating the mistake.

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
