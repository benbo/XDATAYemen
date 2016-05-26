from __future__ import print_function
import json
import sys

import h5py
import numpy as np

import falconn_sims

from flask import Flask, abort, request
app = Flask(__name__)


# based on https://gist.github.com/rossant/7b4704e8caeb8f173084
def _mmap_h5(path, h5path):
    with h5py.File(path, 'r') as f:
        ds = f[h5path]
        # We get the dataset address in the HDF5 file.
        offset = ds.id.get_offset()
        # We ensure we have a non-compressed contiguous array.
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
        dtype = ds.dtype
        shape = ds.shape
    return np.memmap(path, mode='r', shape=shape, offset=offset, dtype=dtype)


def setup(index_filenames, index_features, tune_feats_features,
          target_acc=.9, **kwargs):
    dataset = _mmap_h5(index_features, 'feats')
    tune_queries = _mmap_h5(tune_feats_features, 'feats')

    table, mean = falconn_sims.make_tables(dataset, **kwargs)
    falconn_sims.tune_num_probes(
            table, mean, dataset, tune_queries, target_acc=target_acc)

    with open(index_filenames) as f:
        filenames = [s.strip() for s in f]
    index_lookup = {s.split('/')[1]: i for i, s in enumerate(filenames)}

    def search(feats, k):
        query = feats / np.linalg.norm(feats) - mean
        return table.find_k_nearest_neighbors(query, k)

    def search_filename(filename, k):
        i = index_lookup[filename]
        query = dataset[i] / np.linalg.norm(dataset[i]) - mean
        res = table.find_k_nearest_neighbors(query, k + 1)
        return [filenames[r] for r in res if r != i][:k]

    return dataset.shape[1], search, search_filename


@app.route('/query_filename/<filename>/<int:k>')
def query_filename(filename, k):
    return json.dumps(search_filename(filename, k))


@app.route('/query/<int:k>', methods=['POST'])
def query(k):
    try:
        feats = np.array(json.loads(request.form['features']), dtype=np.float32)
        assert feats.ndim == 1 and feats.shape[0] == dimension
        return json.dumps(search(feats, k))
    except (ValueError, KeyError, AssertionError):
        return abort(400)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('index_filenames')
    parser.add_argument('index_features')
    parser.add_argument('tune_feats_features')
    parser.add_argument('--target-acc', type=float, default=.9)
    parser.add_argument('--already-normed', action='store_true', default=False)
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()

    print("Building indices, will take a while...", file=sys.stderr)
    dimension, search, search_filename = setup(
        args.index_filenames, args.index_features, args.tune_feats_features,
        target_acc=args.target_acc, already_normed=args.already_normed)
    print('done!', file=sys.stderr)

    app.run(host=args.host, port=args.port, debug=True)
