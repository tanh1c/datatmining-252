"""Row-by-row comparison of step 3c vs 0.69972 anchor.

Run AFTER you extract the Colab download into outputs/specter2_finetune_v2/.
Tells you:
- how many rows changed label
- direction of changes (toward higher / lower)
- changes per split (public vs private)
- changes near each threshold (these are the "movable" rows)
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANCHOR = pd.read_csv(ROOT / 'outputs' / '0.69972' / 'specter2_finetune_submission.csv')
NEW = pd.read_csv(ROOT / 'outputs' / '0.68718' / 'specter2_finetune_v2_submission.csv')

merged = ANCHOR.merge(NEW, on='id', suffixes=('_anchor', '_new'))
merged['change'] = merged['Label_new'] - merged['Label_anchor']
merged['changed'] = merged['change'] != 0

print('Total rows:', len(merged))
print(f'Rows changed: {int(merged["changed"].sum())} ({merged["changed"].mean():.1%})')
print()
print('Distribution of changes:')
print(merged.loc[merged["changed"], "change"].value_counts().sort_index().to_string())
print()

# Tag with split
public = pd.read_csv(ROOT / 'data' / 'raw' / 'public_test.csv')[['id']].assign(split='public')
private = pd.read_csv(ROOT / 'data' / 'raw' / 'private_test.csv')[['id']].assign(split='private')
splits = pd.concat([public, private])
merged = merged.merge(splits, on='id', how='left')

print('Changes by split:')
print(merged.groupby('split')['changed'].agg(['count', 'sum', 'mean']).round(3).to_string())
print()
print('Changes by split + direction:')
g = merged[merged['changed']].groupby(['split', 'change']).size().unstack(fill_value=0)
print(g.to_string())
print()
print('First 20 changed rows (sorted by abs change):')
ch = merged[merged['changed']].copy()
ch['abs'] = ch['change'].abs()
ch = ch.sort_values('abs', ascending=False).head(20)

# Add titles
train = pd.read_csv(ROOT / 'data' / 'raw' / 'train.csv')[['id', 'title', 'venue', 'year']]
public_full = pd.read_csv(ROOT / 'data' / 'raw' / 'public_test.csv')[['id', 'title', 'venue', 'year']]
private_full = pd.read_csv(ROOT / 'data' / 'raw' / 'private_test.csv')[['id', 'title', 'venue', 'year']]
all_meta = pd.concat([public_full, private_full])
ch = ch.merge(all_meta, on='id', how='left')

for _, r in ch.iterrows():
    title = str(r['title'])[:80]
    print(f"  [{r['split']:8s}] [{r['venue']} {int(r['year'])}] {r['Label_anchor']} -> {r['Label_new']}  {title}")
