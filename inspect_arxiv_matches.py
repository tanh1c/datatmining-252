"""Inspect arXiv match score distribution to decide if we should lower threshold."""
import pandas as pd
arxiv = pd.read_csv('outputs/external/arxiv_abstracts.csv')
print('Total queries:', len(arxiv))
print('Has any candidate (match_score > 0):', int((arxiv['match_score'] > 0).sum()))
print()
print('Match score distribution:')
print(arxiv['match_score'].describe().round(3).to_string())
print()
print('Score buckets:')
for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.82), (0.82, 0.9), (0.9, 1.0), (1.0, 1.01)]:
    n = int(((arxiv['match_score'] >= lo) & (arxiv['match_score'] < hi)).sum())
    print(f'  [{lo:.2f},{hi:.2f}): {n}')
print()
print('Rows with score in [0.7, 0.82) — would adding these help?')
near = arxiv[(arxiv['match_score'] >= 0.7) & (arxiv['match_score'] < 0.82)]
for _, r in near.head(20).iterrows():
    abs_len = len(str(r['abstract'])) if pd.notna(r['abstract']) else 0
    src = str(r['source_title'])[:60]
    arx = str(r['arxiv_title'])[:60] if pd.notna(r['arxiv_title']) else ''
    score = r['match_score']
    print(f"  score={score:.3f} abs_len={abs_len:4d}  source={src!r}  arxiv={arx!r}")
