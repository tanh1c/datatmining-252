"""Quantify the potential gain from filling in the missing abstracts.

Specifically:
- How many rows are missing per split?
- Are missing rows skewed toward a label? If they are mostly label-1
  proceedings, the model already gets them from the title alone and filling
  abstract won't help much. If they are real research papers (label 2-5),
  filling will help.
- For test rows without abstract, how confident is the current SPECTER2 model?
  Low-confidence predictions are exactly where extra signal could move the
  needle.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = pd.read_csv(ROOT / 'data' / 'raw' / 'train.csv')
PUBLIC = pd.read_csv(ROOT / 'data' / 'raw' / 'public_test.csv')
PRIVATE = pd.read_csv(ROOT / 'data' / 'raw' / 'private_test.csv')
ABS = pd.read_csv(ROOT / 'outputs' / 'external' / 'abstracts_merged_v2.csv')

OOF = pd.read_csv(ROOT / 'outputs' / '0.69972' / 'oof_scores.csv')
PUBLIC_SCORES = pd.read_csv(ROOT / 'outputs' / '0.69972' / 'public_scores.csv')
PRIVATE_SCORES = pd.read_csv(ROOT / 'outputs' / '0.69972' / 'private_scores.csv')

print('='*70)
print('1. Missing abstract counts per split')
print('='*70)
for split in ['train', 'public_test', 'private_test']:
    sub = ABS[ABS['source_split'] == split]
    n_missing = (~sub['has_abstract']).sum()
    print(f'  {split:14s}: {n_missing:4d} / {len(sub):4d} missing  ({n_missing/len(sub):.1%})')

print()
print('='*70)
print('2. Train rows: how do labels distribute when abstract is missing?')
print('='*70)
train_abs = ABS[ABS['source_split']=='train'][['id','has_abstract','abstract_source']]
t = TRAIN.merge(train_abs, on='id', how='left')
t['has_abstract'] = t['has_abstract'].fillna(False)
print(f'  Train baseline label dist:')
print('   ', dict(t['Label'].value_counts(normalize=True).sort_index().round(3)))
print(f'  Train missing-abstract rows ({(~t["has_abstract"]).sum()}):')
print('   ', dict(t.loc[~t['has_abstract'], 'Label'].value_counts(normalize=True).sort_index().round(3)))
print(f'  Train with-abstract rows ({t["has_abstract"].sum()}):')
print('   ', dict(t.loc[t['has_abstract'], 'Label'].value_counts(normalize=True).sort_index().round(3)))

print()
print('='*70)
print('3. Train: OOF QWK on missing-abstract subset vs with-abstract subset')
print('='*70)
t = t.merge(OOF[['id','oof_score','oof_pred']], on='id', how='left')
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
def qwk(d):
    if len(d) < 5:
        return float('nan'), float('nan')
    return (cohen_kappa_score(d['Label'], d['oof_pred'], weights='quadratic'),
            mean_absolute_error(d['Label'], d['oof_pred']))
qwk_with, mae_with = qwk(t[t['has_abstract']])
qwk_without, mae_without = qwk(t[~t['has_abstract']])
print(f'  with abstract ({t["has_abstract"].sum()} rows):    OOF QWK = {qwk_with:.4f}, MAE = {mae_with:.3f}')
print(f'  without abstract ({(~t["has_abstract"]).sum()} rows): OOF QWK = {qwk_without:.4f}, MAE = {mae_without:.3f}')
print(f'  ==> gap = {qwk_with - qwk_without:+.4f}')

print()
print('='*70)
print('4. Test rows without abstract: confidence of SPECTER2 score?')
print('='*70)
pub_abs = ABS[ABS['source_split']=='public_test'][['id','has_abstract']]
priv_abs = ABS[ABS['source_split']=='private_test'][['id','has_abstract']]
pub = PUBLIC.merge(pub_abs, on='id').merge(PUBLIC_SCORES, on='id', how='left')
priv = PRIVATE.merge(priv_abs, on='id').merge(PRIVATE_SCORES, on='id', how='left')
pub['has_abstract'] = pub['has_abstract'].fillna(False)
priv['has_abstract'] = priv['has_abstract'].fillna(False)

# A "confident" prediction is one whose score is >0.4 away from the nearest
# threshold. The 0.69972 thresholds are [1.81, 2.46, 3.23, 4.13].
THRESH = np.array([1.808, 2.457, 3.230, 4.132])
def confidence(score):
    return np.min(np.abs(score - THRESH))

for name, df in [('public', pub), ('private', priv)]:
    miss = df[~df['has_abstract']]
    have = df[df['has_abstract']]
    miss_conf = miss['score'].map(confidence)
    have_conf = have['score'].map(confidence)
    print(f'  {name}: missing-abstract rows confidence (lower = closer to threshold = riskier)')
    print(f'    mean confidence: missing={miss_conf.mean():.3f}, with_abstract={have_conf.mean():.3f}')
    print(f'    rows within 0.2 of nearest threshold: missing={int((miss_conf<0.2).sum())}/{len(miss)}, '
          f'with_abstract={int((have_conf<0.2).sum())}/{len(have)}')

print()
print('='*70)
print('5. Test rows without abstract: which venues / years?')
print('='*70)
for name, df in [('public', pub), ('private', priv)]:
    miss = df[~df['has_abstract']]
    print(f'  {name}: {len(miss)} rows missing abstract')
    print(f'    venues: {dict(miss["venue"].value_counts())}')
    print(f'    years: {dict(miss["year"].value_counts().sort_index())}')

print()
print('='*70)
print('6. Sample missing-abstract test titles (do they look "real" or "proceedings"?)')
print('='*70)
for name, df in [('public', pub), ('private', priv)]:
    miss = df[~df['has_abstract']].head(8)
    print(f'  {name} sample missing:')
    for _, r in miss.iterrows():
        title = str(r['title'])[:90]
        print(f'    [{r["venue"]} {int(r["year"])}] {title}  (score={r["score"]:.2f})')

print()
print('='*70)
print('7. Coarse upper-bound QWK gain estimate')
print('='*70)
# Suppose filling abstract pulls each missing test prediction by `delta_label`
# closer to truth on average. Calibrated by the train OOF gap.
n_missing_test = (~pub['has_abstract']).sum() + (~priv['has_abstract']).sum()
n_total_test = len(pub) + len(priv)
print(f'  Total test rows missing abstract: {n_missing_test}/{n_total_test} = {n_missing_test/n_total_test:.1%}')

# Train-side QWK gap suggests label MAE on missing rows is ~mae_without vs mae_with.
mae_gap = mae_without - mae_with
print(f'  Train MAE gap (missing - with) = {mae_gap:+.3f}')
print(f'  If we could close the gap fully on the {n_missing_test} test rows:')
print(f'    expected MAE reduction across whole test = {n_missing_test/n_total_test * mae_gap:.4f}')
print(f'    rough QWK gain estimate (since QWK ~ 1 - MAE/MAE_baseline): +0.005 to +0.015')
