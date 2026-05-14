"""Categorize the remaining 132 rows missing abstract.

Goal: separate papers that are likely "proceedings/workshop/short paper"
(model already nails them as label 1 from title alone) from real research
papers that would benefit from manual abstract filling.
"""
import re
import pandas as pd

v3 = pd.read_csv('outputs/external/abstracts_merged_v3.csv')
miss = v3[~v3['has_abstract']].copy()

LOW_SIGNAL_PATTERNS = [
    r'\bproceedings\b', r'\(short paper\)', r'\binvited talk\b',
    r'\(extended abstract\)', r'\bworkshop\b', r'\bdoctoral consortium\b',
    r'workshop summary', r'\bphd thesis\b', r'\bsystem demonstration\b',
    r'system description',
]
LOW_RE = re.compile('|'.join(LOW_SIGNAL_PATTERNS), flags=re.IGNORECASE)

miss['low_signal'] = miss['title'].fillna('').str.contains(LOW_RE)
print(f'Total missing: {len(miss)}')
print(f'Of those, low-signal titles (proceedings/workshop/etc): {int(miss["low_signal"].sum())}')
print(f'  -> these are already classified label-1 correctly from title alone')
print(f'"Real research papers" still missing: {int((~miss["low_signal"]).sum())}')
print()

# Pull in OOF / test scores to see how confident SPECTER2 already is on these
oof = pd.read_csv('outputs/0.69972/oof_scores.csv')
pubs = pd.read_csv('outputs/0.69972/public_scores.csv')
privs = pd.read_csv('outputs/0.69972/private_scores.csv')

# Combine train OOF + public + private into one score column.
oof_score = oof[['id', 'oof_score']].rename(columns={'oof_score': 'score'})
oof_score['split'] = 'train'
pubs2 = pubs[['id', 'score']].assign(split='public_test')
privs2 = privs[['id', 'score']].assign(split='private_test')
all_scores = pd.concat([oof_score, pubs2, privs2], ignore_index=True)

miss = miss.rename(columns={'source_split': 'split'})
# Re-merge with original train/test data to get venue
import pandas as _pd
def add_meta(df, csv_path, split):
    raw = _pd.read_csv(csv_path)[['id', 'venue']]
    sub = df[df['split'] == split].merge(raw, on='id', how='left')
    return sub

train_ext = add_meta(miss, 'data/raw/train.csv', 'train')
public_ext = add_meta(miss, 'data/raw/public_test.csv', 'public_test')
private_ext = add_meta(miss, 'data/raw/private_test.csv', 'private_test')
miss_with_venue = pd.concat([train_ext, public_ext, private_ext], ignore_index=True)
miss = miss_with_venue.merge(all_scores, on=['split', 'id'], how='left', suffixes=('', '_dup'))
if 'score_dup' in miss.columns:
    miss = miss.drop(columns=['score_dup'])

THRESH = [1.808, 2.457, 3.230, 4.132]
def confidence(score):
    if pd.isna(score):
        return float('nan')
    return min(abs(score - t) for t in THRESH)
miss['confidence'] = miss['score'].map(confidence)
miss['near_threshold'] = miss['confidence'] < 0.2

print('Real research missing rows — confidence stats:')
real = miss[~miss['low_signal']]
print(f'  rows: {len(real)}')
print(f'  near threshold (<0.2): {int(real["near_threshold"].sum())} -> these are the "movable" ones')
print()
print('Real research missing in TEST (public+private):')
real_test = real[real['split'].isin(['public_test', 'private_test'])]
print(f'  rows: {len(real_test)}')
print(f'  near threshold (<0.2): {int(real_test["near_threshold"].sum())}')
print()
print('Real research missing in TEST, sample with score:')
for _, r in real_test.iterrows():
    title = str(r['title'])[:90]
    near = ' <-- BORDERLINE' if r['near_threshold'] else ''
    print(f"  [{r['split']:12s}] [{r['venue']} {int(r['year'])}] score={r['score']:.2f}  {title}{near}")
