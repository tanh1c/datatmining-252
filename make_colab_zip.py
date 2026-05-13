"""Zip the 5 input files into asp_data.zip for upload to Google Colab.

Run:
    python make_colab_zip.py

Output: asp_data.zip (in project root) containing:
    train.csv
    public_test.csv
    private_test.csv
    Test_Submission.csv
    abstracts_merged_v2.csv
"""
from __future__ import annotations
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = [
    ROOT / 'data' / 'raw' / 'train.csv',
    ROOT / 'data' / 'raw' / 'public_test.csv',
    ROOT / 'data' / 'raw' / 'private_test.csv',
    ROOT / 'data' / 'raw' / 'Test_Submission.csv',
    ROOT / 'outputs' / 'external' / 'abstracts_merged_v2.csv',
]
for f in files:
    if not f.exists():
        raise SystemExit(f'missing: {f}')

zip_path = ROOT / 'asp_data.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, arcname=f.name)

size_mb = zip_path.stat().st_size / 1e6
print(f'wrote {zip_path}  ({size_mb:.2f} MB)')
print('upload this to the Colab notebook (cell 4).')
