"""Convert the SP-TransientBench histogram text files of one room to arrays.

Example:
    python scripts/prepare_rooms.py --src data/sp_transientbench/c4floor2 \
        --out data/sp_transientbench/hist/c4floor2
writes raw_NN.npy (uint16, 192x256x672) per view, numbered like the Livox files.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from firstcross.data_rooms import parse_histogram  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', required=True, help='room folder with sp_pose_results.csv')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.src, 'sp_pose_results.csv')) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        k = int(r['livox_csv_name'].split('.')[0])
        dst = os.path.join(args.out, f'raw_{k:02d}.npy')
        if not os.path.exists(dst):
            np.save(dst, parse_histogram(os.path.join(args.src, r['sp_file'])))
    print(f'{len(rows)} views in {args.out}')


if __name__ == '__main__':
    main()
