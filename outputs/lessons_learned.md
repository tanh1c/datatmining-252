# Lessons learned — for the next agent / future-me

Living document. Each entry records a concrete observation we paid for in
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
