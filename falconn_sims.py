from __future__ import division, print_function
from functools import partial
import sys

import falconn
import numpy as np

def make_tables(dataset, num_queries=1000, num_tables=50, copy=True,
                seed=None, num_threads=0, verbose=True):
    p = partial(print, file=sys.stderr) if verbose else lambda *a, **kw: None
    norms = np.linalg.norm(dataset, axis=1)
    if copy:
        dataset = dataset / norms[:, np.newaxis]
    else:
        dataset /= norms[:, np.newaxis]

    normed_mean = dataset.mean(axis=0)
    dataset -= normed_mean

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = dataset.shape[1]
    params_cp.lsh_family = 'cross_polytope'
    params_cp.distance_function = 'euclidean_squared'
    params_cp.l = num_tables
    params_cp.num_rotations = 1  # try 2, maybe
    params_cp.seed = seed if seed is not None else np.random.randint(2**31)
    params_cp.num_setup_threads = num_threads
    params_cp.storage_hash_table = 'bit_packed_flat_hash_table'
    n_bits = int(np.round(np.log2(dataset.shape[0])))
    falconn.compute_number_of_hash_functions(n_bits, params_cp)

    p('Starting building table...', end='')
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    p('done')

    return table, normed_mean


def tune_num_probes(table, mean, dataset, queries, answers=None, verbose=True):
    p = partial(print, file=sys.stderr) if verbose else lambda *a, **kw: None
    
    queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
    queries -= mean

    if answers is None:
        p('Getting true answers to queries...', end='')
        answers = np.dot(dataset, queries.T).argmax(axis=0)
        p('done.')

    def eval_num_probes(num_probes):
        table.set_num_probes(num_probes)
        score = 0
        for (i, query) in enumerate(queries):
            if answers[i] in table.get_candidates_with_duplicates(query):
                score += 1
        return score / len(queries)

    num_probes = table.get_num_probes()
    stepped = False
    while True:
        acc = eval_num_probes(num_probes)
        p('{} -> {}'.format(num_probes, acc))
        if acc >= .9:
            break
        num_probes *= 2
        stepped = True

    if stepped:
        left = num_probes // 2
        right = num_probes
        while right - left > 1:
            num_probes = (left + right) // 2
            acc = eval_num_probes(num_probes)
            p('{} -> {}'.format(num_probes, acc))
            if acc >= .9:
                right = num_probes
            else:
                left = num_probes
        num_probes = right

    table.set_num_probes(num_probes)
