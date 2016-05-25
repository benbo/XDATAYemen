from __future__ import division, print_function
import sys

from annoy import AnnoyIndex
import h5py
import progressbar as pb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('dataset')
parser.add_argument('out_file')
parser.add_argument('--trees', type=int, default=256)
args = parser.parse_args()

with h5py.File(args.filename, 'r') as f:
    X = f[args.dataset]
    idx = AnnoyIndex(X.shape[1], 'angular')
    print("Adding items...", file=sys.stderr, end='')
    idx.add_item(X.shape[0] - 1, X[-1])  # preallocate full size
    for i, v in enumerate(pb.ProgressBar()(X[:-1])):
        idx.add_item(i, v)
    print("done.", file=sys.stderr)

print("Building trees...", file=sys.stderr, end='')
idx.build(args.trees)
print("done.", file=sys.stderr)

print("Saving index...", file=sys.stderr, end='')
idx.save(args.out_file)
print("done.", file=sys.stderr)
