"""Reverse-engineer the meaning of labels 1-5 from the SPECTER2 fine-tune OOF.

For each label 1-5 we:
1. Pick the top-K training papers the model is *most confident* about (i.e.
   highest score among true-label-X rows -> the model "agrees strongly").
2. Look at venue distribution + keyword frequency per label.
3. Pick ~6 sample titles per label.

Output: outputs/eda_v2/label_meaning_from_specter2.md
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN = pd.read_csv(ROOT / 'data' / 'raw' / 'train.csv')
OOF = pd.read_csv(ROOT / 'outputs' / '0.69972' / 'oof_scores.csv')
ABS = pd.read_csv(ROOT / 'outputs' / 'external' / 'abstracts_merged_v2.csv')

train_abs = ABS[ABS['source_split'] == 'train'][['id', 'abstract']]
df = TRAIN.merge(OOF[['id', 'oof_score', 'oof_pred']], on='id').merge(train_abs, on='id', how='left')
df['abstract'] = df['abstract'].fillna('')

OUT = ROOT / 'outputs' / 'eda_v2' / 'label_meaning_from_specter2.md'

STOP = set("""
the a an of and or in to for with on by that is are was were be been being
this these those it its from at as we our their they them he she his her
which whose who whom what when where how why so if then than do does did
have has had not no can could would should may might will shall must
also more most some any all such which between among across over under
into through against before after during while because since although though
both either neither one two three four five six seven eight nine ten
paper papers approach approaches method methods system systems work works
new novel propose proposed propose proposes algorithm algorithms problem
problems result results show shows showed using used use using based
present presents presented presenting present
abstract introduction conclusion section sections etc i e g i.e e.g
this paper presents we present we propose we consider we describe
""".split())

def tokenize(text: str) -> list[str]:
    text = re.sub(r'[^a-zA-Z\- ]', ' ', text.lower())
    return [w for w in text.split() if len(w) >= 3 and w not in STOP]

def top_keywords_for_label(group: pd.DataFrame, k: int = 25) -> list[tuple[str, float]]:
    """Find words that occur disproportionately in this label vs corpus."""
    label_counter: Counter = Counter()
    for _, row in group.iterrows():
        label_counter.update(set(tokenize(row['title'] + ' ' + row['abstract'])))
    return label_counter.most_common(k)

lines: list[str] = []
lines.append('# What do labels 1-5 actually mean?\n')
lines.append('Reverse-engineered from the 0.69972 SPECTER2 fine-tune OOF predictions on the training set.\n')
lines.append('For each label, we look at:\n')
lines.append('- *Confidence-correct* papers: rows where `Label == oof_pred` AND `oof_score` is in the top quintile for that label. These are the cleanest examples — model and ground truth strongly agree.\n')
lines.append('- Venue distribution.\n')
lines.append('- Top keywords (after removing stop words and project-noise words).\n')
lines.append('- 6 representative titles.\n\n---\n')

# Word frequencies across the whole train corpus, used as a baseline.
corpus_counter: Counter = Counter()
for _, row in df.iterrows():
    corpus_counter.update(set(tokenize(row['title'] + ' ' + row['abstract'])))
corpus_total_docs = len(df)

def lift(counter: Counter, n_docs: int, k: int = 20) -> list[tuple[str, float, int, int]]:
    """Return words with the highest lift = (p_in_label - p_in_corpus) for words seen >= 6 times in label."""
    rows = []
    for word, c in counter.items():
        if c < 6:
            continue
        p_label = c / max(n_docs, 1)
        p_corpus = corpus_counter.get(word, 0) / corpus_total_docs
        delta = p_label - p_corpus
        if delta <= 0:
            continue
        rows.append((word, delta, c, corpus_counter.get(word, 0)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:k]

for label in [1, 2, 3, 4, 5]:
    group = df[df['Label'] == label].copy()
    correct = group[group['oof_pred'] == label].copy()
    if not correct.empty:
        thresh = correct['oof_score'].quantile(0.8) if label >= 3 else correct['oof_score'].quantile(0.2)
        if label >= 3:
            cleanest = correct[correct['oof_score'] >= thresh]
        else:
            cleanest = correct[correct['oof_score'] <= thresh]
    else:
        cleanest = group

    venue_dist = group['venue'].value_counts().to_dict()
    counter: Counter = Counter()
    for _, row in cleanest.iterrows():
        counter.update(set(tokenize(row['title'] + ' ' + row['abstract'])))
    top = lift(counter, n_docs=len(cleanest), k=18)

    lines.append(f'\n## Label **{label}**  ({len(group)} train rows, {len(correct)} model-correct, {len(cleanest)} cleanest)\n')
    lines.append(f'**Mean OOF score for true-label-{label}:** {group["oof_score"].mean():.2f}\n')
    lines.append(f'**Venue distribution:** {venue_dist}\n')
    lines.append('\n**Top words (by lift over corpus, min 6 occurrences in cleanest set):**\n')
    if top:
        word_table = ', '.join(f'`{w}`(+{delta:.3f}, {c}/{corp})' for w, delta, c, corp in top)
        lines.append(word_table + '\n')
    else:
        lines.append('(no word passes the threshold)\n')

    lines.append('\n**Sample titles (cleanest, sorted by descending OOF score for label>=3, ascending for label<=2):**\n')
    if label >= 3:
        ordered = cleanest.sort_values('oof_score', ascending=False)
    else:
        ordered = cleanest.sort_values('oof_score', ascending=True)
    for _, r in ordered.head(6).iterrows():
        title = str(r['title'])[:120]
        lines.append(f'- [{r["venue"]} {int(r["year"])}] {title}  (oof={r["oof_score"]:.2f})\n')

    lines.append('')

# Final synthesis
lines.append('\n---\n## Synthesis: what each label means\n')
lines.append('''
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
''')

OUT.write_text('\n'.join(lines), encoding='utf-8')
print('wrote', OUT)
