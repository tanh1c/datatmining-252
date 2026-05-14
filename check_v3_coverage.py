"""Quick coverage check on abstracts_merged_v3."""
import pandas as pd
v3 = pd.read_csv('outputs/external/abstracts_merged_v3.csv')
miss = v3[~v3['has_abstract']]
print(f'Total rows in v3: {len(v3)}')
print(f'Has abstract: {int(v3["has_abstract"].sum())} ({v3["has_abstract"].mean():.1%})')
print(f'Still missing: {len(miss)} rows\n')
print('Missing by split:')
print(miss.groupby('source_split').size().to_string())
print('\nSample missing titles (first 12):')
for _, r in miss.head(12).iterrows():
    title = str(r['title'])[:90]
    split = r['source_split']
    year = int(r['year']) if pd.notna(r['year']) else 0
    print(f"  [{split:14s}] [{year}] {title}")
print('\nSource breakdown for found rows:')
print(v3.groupby(['source_split', 'abstract_source']).size().unstack(fill_value=0).to_string())
