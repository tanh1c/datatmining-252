"""Quick coverage and quality check for the merged abstract cache."""
import pandas as pd

df = pd.read_csv('outputs/external/abstracts_merged_v2.csv')
print('Total rows:', len(df))
print()
print('Coverage by split:')
print(df.groupby('source_split')['has_abstract'].agg(['count', 'sum', 'mean']).round(3))
print()
print('Source breakdown (first non-empty):')
print(df.groupby(['source_split', 'abstract_source']).size().unstack(fill_value=0))
print()
print('Abstract length stats (rows with has_abstract=True):')
print(df.loc[df['has_abstract'], 'abstract_len'].describe().round(1))
print()
print('Sample abstracts:')
sample = df[df['has_abstract']].sample(3, random_state=42)
for _, r in sample.iterrows():
    print(f"[{r['abstract_source']}] {str(r['title'])[:80]}")
    print(f"  -> {str(r['abstract'])[:240]}...")
    print()
