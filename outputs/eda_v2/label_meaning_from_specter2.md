# What do labels 1-5 actually mean?

Reverse-engineered from the 0.69972 SPECTER2 fine-tune OOF predictions on the training set.

For each label, we look at:

- *Confidence-correct* papers: rows where `Label == oof_pred` AND `oof_score` is in the top quintile for that label. These are the cleanest examples — model and ground truth strongly agree.

- Venue distribution.

- Top keywords (after removing stop words and project-noise words).

- 6 representative titles.

---


## Label **1**  (903 train rows, 575 model-correct, 115 cleanest)

**Mean OOF score for true-label-1:** 1.75

**Venue distribution:** {'cav': 300, 'lics': 288, 'kr': 177, 'iclp': 117, 'lpnmr': 21}


**Top words (by lift over corpus, min 6 occurrences in cleanest set):**

`proceedings`(+0.349, 43/62), `conference`(+0.339, 42/65), `international`(+0.326, 41/75), `july`(+0.271, 33/40), `cav`(+0.203, 25/35), `computer`(+0.186, 25/78), `aided`(+0.140, 17/20), `part`(+0.125, 18/79), `workshop`(+0.113, 14/23), `usa`(+0.106, 13/17), `iclp`(+0.105, 13/19), `symposium`(+0.098, 12/16), `verification`(+0.093, 24/289), `ieee`(+0.088, 11/18), `september`(+0.081, 10/16), `germany`(+0.074, 9/11), `acm`(+0.072, 9/15), `co-located`(+0.071, 9/17)


**Sample titles (cleanest, sorted by descending OOF score for label>=3, ascending for label<=2):**

- [iclp 2026] Proceedings 41st International Conference on Logic Programming, ICLP 2025, Rende, Italy, 12-19th September 2025.  (oof=1.00)

- [iclp 2025] Proceedings of the 9th Workshop on Advances in Argumentation in Artificial Intelligence (AI3 2025) co-located with the 4  (oof=1.00)

- [iclp 2023] Proceedings 39th International Conference on Logic Programming, ICLP 2023, Imperial College London, UK, 9th July 2023 -   (oof=1.00)

- [iclp 2024] Workshop Proceedings of the 40th International Conference on Logic Programming (ICLP-WS 2024) co-located with the 40th I  (oof=1.00)

- [iclp 2019] Proceedings 35th International Conference on Logic Programming (Technical Communications), ICLP 2019 Technical Communica  (oof=1.00)

- [kr 2025] Emerson-Lei and Manna-Pnueli Games for LTLf+ and PPLTL+ Synthesis.  (oof=1.00)



## Label **2**  (514 train rows, 133 model-correct, 27 cleanest)

**Mean OOF score for true-label-2:** 2.29

**Venue distribution:** {'lics': 149, 'cav': 135, 'kr': 134, 'iclp': 78, 'lpnmr': 18}


**Top words (by lift over corpus, min 6 occurrences in cleanest set):**

`checking`(+0.217, 8/199), `program`(+0.169, 7/225), `verification`(+0.106, 6/289), `first`(+0.070, 6/380), `model`(+0.042, 7/541)


**Sample titles (cleanest, sorted by descending OOF score for label>=3, ascending for label<=2):**

- [cav 2016] From Shape Analysis to Termination Analysis in Linear Time.  (oof=1.82)

- [lics 2022] Monoidal Streams for Dataflow Programming.  (oof=1.82)

- [cav 2022] Verified Numerical Methods for Ordinary Differential Equations.  (oof=1.82)

- [cav 2023] nekton: A Linearizability Proof Checker.  (oof=1.82)

- [lpnmr 2019] Repair-Based Degrees of Database Inconsistency.  (oof=1.83)

- [kr 2025] Explanations for Unrealizability of Infinite-State Safety Shields.  (oof=1.83)



## Label **3**  (438 train rows, 124 model-correct, 25 cleanest)

**Mean OOF score for true-label-3:** 2.63

**Venue distribution:** {'kr': 132, 'cav': 109, 'lics': 99, 'iclp': 75, 'lpnmr': 23}


**Top words (by lift over corpus, min 6 occurrences in cleanest set):**

`reasoning`(+0.187, 9/432), `models`(+0.134, 7/364), `given`(+0.089, 6/376), `logic`(+0.087, 9/681)


**Sample titles (cleanest, sorted by descending OOF score for label>=3, ascending for label<=2):**

- [iclp 2025] A Scalable Approach to Probabilistic Compliance in Declarative Process Mining.  (oof=3.23)

- [lics 2016] Differential Refinement Logic.  (oof=3.21)

- [kr 2016] Quantifying Conflicts for Spatial and Temporal Information.  (oof=3.20)

- [cav 2023] Rounding Meets Approximate Model Counting.  (oof=3.19)

- [kr 2025] Guarded Fragments Meet Dynamic Logic: The Story of Regular Guards.  (oof=3.19)

- [cav 2021] Model-Free Reinforcement Learning for Branching Markov Decision Processes.  (oof=3.18)



## Label **4**  (367 train rows, 122 model-correct, 25 cleanest)

**Mean OOF score for true-label-4:** 3.34

**Venue distribution:** {'kr': 127, 'iclp': 86, 'cav': 69, 'lics': 53, 'lpnmr': 32}


**Top words (by lift over corpus, min 6 occurrences in cleanest set):**

`answer`(+0.207, 8/283), `multiple`(+0.197, 6/108), `set`(+0.195, 10/511), `reasoning`(+0.187, 9/432), `programming`(+0.184, 9/440), `language`(+0.174, 8/365), `asp`(+0.160, 6/199), `knowledge`(+0.152, 7/319), `study`(+0.096, 6/359), `given`(+0.089, 6/376)


**Sample titles (cleanest, sorted by descending OOF score for label>=3, ascending for label<=2):**

- [iclp 2024] Causally Constrained Counterfactual Generation using ASP.  (oof=4.12)

- [iclp 2025] Flexible, Lifelong, Explainable, and Robust Solutions for Multi-Agent Path Finding Problems.  (oof=4.11)

- [lics 2017] Symbolic execution and probabilistic reasoning.  (oof=4.09)

- [kr 2020] An Answer Set Programming Approach to Argumentative Reasoning in the ASPIC+ Framework.  (oof=4.08)

- [lpnmr 2019] Treewidth and Counting Projected Answer Sets.  (oof=4.08)

- [lpnmr 2024] Answer Set Explanations via Preferred Unit-Provable Unsatisfiable Subsets.  (oof=4.08)



## Label **5**  (272 train rows, 140 model-correct, 28 cleanest)

**Mean OOF score for true-label-5:** 3.96

**Venue distribution:** {'iclp': 100, 'kr': 91, 'cav': 44, 'lpnmr': 22, 'lics': 15}


**Top words (by lift over corpus, min 6 occurrences in cleanest set):**

`answer`(+0.565, 19/283), `programming`(+0.466, 18/440), `set`(+0.438, 18/511), `asp`(+0.420, 14/199), `neural`(+0.313, 10/109), `networks`(+0.302, 10/137), `explanations`(+0.294, 9/68), `explainable`(+0.266, 8/48), `deep`(+0.262, 8/59), `performance`(+0.243, 9/195), `learning`(+0.233, 9/220), `explain`(+0.232, 7/46), `however`(+0.221, 9/251), `reasoning`(+0.220, 11/432), `models`(+0.211, 10/364), `tasks`(+0.205, 7/113), `intelligence`(+0.197, 6/44), `artificial`(+0.195, 6/48)


**Sample titles (cleanest, sorted by descending OOF score for label>=3, ascending for label<=2):**

- [iclp 2025] Formally Explaining Decision Tree Models with Answer Set Programming.  (oof=5.00)

- [iclp 2025] xDNN(ASP): Explanation Generation System for Deep Neural Networks powered by Answer Set Programming.  (oof=5.00)

- [iclp 2023] Explanations for Answer Set Programming.  (oof=5.00)

- [iclp 2023] Reliable Natural Language Understanding with Large Language Models and Answer Set Programming.  (oof=5.00)

- [iclp 2022] Tools and Methodologies for Verifying Answer Set Programs.  (oof=5.00)

- [iclp 2023] Explainable Answer-set Programming.  (oof=5.00)



---
## Synthesis: what each label means


Based on the keyword + venue + sample-title patterns above, here is the most
defensible interpretation of the labels:

- **Label 1 — Off-topic / non-research / generic logic**: proceedings volumes,
  invited talks, short papers, doctoral-consortium / workshop papers, and
  papers in adjacent fields (verification, type theory, generic logic) that
  have *no ASP component*. These are the "clearly not ASP" papers.

- **Label 2 — Loose adjacency to ASP / KR**: knowledge representation,
  argumentation, description logic, default reasoning, planning. Logic-flavoured
  but the core method is not Answer Set Programming. The text mentions logic
  primitives but not solvers / encodings / ASP-specific constructs.

- **Label 3 — Generic logic / declarative reasoning that touches ASP**:
  Prolog, Datalog, abductive reasoning, NMR, qualitative reasoning, epistemic
  logic. Often share a vocabulary with ASP (rules, semantics, fixpoints) but
  the contribution is at the meta or generic logic-programming layer.

- **Label 4 — Applied / extending ASP**: papers that *use* ASP to solve a
  domain problem, *extend* ASP with new constructs (probabilities, choice,
  preferences, learning, MAS), or build tools / encodings on top of an ASP
  solver. The ASP component is central but not the only contribution.

- **Label 5 — Core ASP advances or ASP+AI cross-overs**: ASP solver/grounder
  algorithms (Clingo, DLV, ASP(Q)), formal semantics of ASP, ASP-driven
  learning of programs/heuristics, neural-symbolic systems built on top of
  ASP. *Inside CAV/LICS*, label 5 also covers neural-network verification and
  modern ML-meets-formal-methods work — i.e. "AI-symbolic" papers regardless
  of venue. This is the strongest signal direction the model picks up.

In one sentence: **Label = how central are Answer Set Programming and the
modern AI-symbolic (neural verification, neuro-symbolic, ML-meets-logic)
agenda to the paper.**
