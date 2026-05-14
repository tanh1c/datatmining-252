"""Build asp_data_v3.zip with the v3 abstract cache for the next Colab run."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = [
    ROOT / 'data' / 'raw' / 'train.csv',
    ROOT / 'data' / 'raw' / 'public_test.csv',
    ROOT / 'data' / 'raw' / 'private_test.csv',
    ROOT / 'data' / 'raw' / 'Test_Submission.csv',
    ROOT / 'outputs' / 'external' / 'abstracts_merged_v3.csv',
]
for f in files:
    if not f.exists():
        raise SystemExit(f'missing: {f}')

zip_path = ROOT / 'asp_data_v3.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, arcname=f.name)
print(f'wrote {zip_path}  ({zip_path.stat().st_size/1e6:.2f} MB)')
print('upload this to the Colab notebook (cell 4).')
