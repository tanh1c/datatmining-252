"""Find the strongest few-shot example candidates per label.

Criteria: (1) train rows where SPECTER2 OOF prediction matches the true label,
(2) abstract is non-empty (so we have something to point at),
(3) the title mentions explicit ASP/AI-symbolic vocabulary that we can quote
in the rationale.

Output: outputs/eda_v2/few_shot_candidates.md
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN = pd.read_csv(ROOT / 'data' / 'raw' / 'train.csv')
SPECTER_OOF = pd.read_csv(ROOT / 'outputs' / '0.69972' / 'oof_scores.csv')
ABS = pd.read_csv(ROOT / 'outputs' / 'external' / 'abstracts_merged_v2.csv')
LLM_OOF_PATH = ROOT / 'outputs' / 'llm_zeroshot' / 'oof_scores.csv'
LLM_OOF = pd.read_csv(LLM_OOF_PATH) if LLM_OOF_PATH.exists() else None

train_abs = ABS[ABS['source_split'] == 'train'][['id', 'abstract', 'has_abstract']]
df = TRAIN.merge(SPECTER_OOF[['id', 'oof_score', 'oof_pred']], on='id') \
         .merge(train_abs, on='id', how='left')
df['abstract'] = df['abstract'].fillna('')
if LLM_OOF is not None:
    llm = LLM_OOF[['id', 'oof_score', 'oof_pred']].rename(columns={'oof_score': 'llm_score', 'oof_pred': 'llm_pred'})
    df = df.merge(llm, on='id', how='left')

ASP_KW = re.compile(r'\b(asp|answer set|stable model|clingo|dlv|aspq|grounder)\b', re.IGNORECASE)
NEURO_KW = re.compile(r'\b(neural|neuro-?symbolic|deep learning|llm|transformer|explainab)\b', re.IGNORECASE)
LOW_KW = re.compile(r'\b(proceedings|workshop|short paper|invited|extended abstract)\b', re.IGNORECASE)

print('### Few-shot candidates per label\n')
for label in [1, 2, 3, 4, 5]:
    print(f'\n## Label {label}')
    sub = df[(df['Label'] == label) & (df['oof_pred'] == label) & (df['has_abstract'])].copy()

    # Score a "rationale clarity" rank
    def cue_score(row):
        text = (str(row['title']) + ' ' + str(row['abstract'])).lower()
        cues = 0
        if label >= 4 and ASP_KW.search(text): cues += 2
        if label == 5 and NEURO_KW.search(text): cues += 1
        if label == 1 and LOW_KW.search(row['title']): cues += 2
        if label == 2 and re.search(r'\b(verification|hyperproperty|model checking)\b', text): cues += 1
        if label == 3 and re.search(r'\b(datalog|prolog|reasoning|epistemic|argumentation)\b', text): cues += 1
        # Prefer SPECTER2-confident papers
        return cues + abs(row['oof_score'] - label) * -1.0

    sub['cue_score'] = sub.apply(cue_score, axis=1)
    sub = sub.sort_values('cue_score', ascending=False)
    for _, r in sub.head(6).iterrows():
        print(f"\nid={int(r['id'])}  venue={r['venue']}  year={int(r['year'])}  specter_oof={r['oof_score']:.2f}  cue_score={r['cue_score']:.2f}")
        if 'llm_score' in df.columns and not pd.isna(r.get('llm_score')):
            print(f"  llm_score={r['llm_score']:.2f}")
        print(f"  TITLE: {str(r['title'])[:140]}")
        ab = str(r['abstract'])[:380]
        print(f"  ABSTRACT: {ab}")
