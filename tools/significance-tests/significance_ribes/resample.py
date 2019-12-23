# -*- coding: utf-8 -*-
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()

    # Translation files
    parser.add_argument('reference_dir', type=str, help='path to reference translations file')
    parser.add_argument('system_x_dir', type=str, help='path to system X translations file')
    parser.add_argument('system_y_dir', type=str, help='path to system Y translations file')

    # General options
    parser.add_argument('--seed', type=int, default=1234, help='pseudo random number generator seed')

    # Significance tests options
    parser.add_argument('--num-bootstrap-samples', '-B', type=int, default=1000, help='number of bootstreap samples')
    parser.add_argument('--sample-size', '-S', type=int, default=None, help='number of sentences per sample')

    return parser


if __name__ == '__main__':

    # Init
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Read translation files
    with open(args.reference_dir, 'r') as f:
        ref_translations = np.array(f.readlines())
    with open(args.system_x_dir, 'r') as f:
        x_translations = np.array(f.readlines())
    with open(args.system_y_dir, 'r') as f:
        y_translations = np.array(f.readlines())
    n_refs, n_xs, n_ys = len(ref_translations), len(x_translations), len(y_translations)
    assert n_refs == n_xs == n_ys, 'Different number of sentences in given files: %d, %d, %d' % (n_refs, n_xs, n_ys)

    sample_size = n_refs if args.sample_size is None else args.sample_size
    sample_ixs = [np.random.choice(range(n_refs), sample_size) for _ in range(args.num_bootstrap_samples)]

    for i, ixs in enumerate(sample_ixs):
        refs_ixs = ref_translations[ixs]
        xs_ixs = x_translations[ixs]
        ys_ixs = y_translations[ixs]
        with open('refs.'+str(i), 'w') as f:
            for line in refs_ixs:
                line = line[:-1] if line[-1] == '\n' else line
                f.write(line + '\n')
        with open('xs.'+str(i), 'w') as f:
            for line in xs_ixs:
                line = line[:-1] if line[-1] == '\n' else line
                f.write(line + '\n')
        with open('ys.'+str(i), 'w') as f:
            for line in ys_ixs:
                line = line[:-1] if line[-1] == '\n' else line
                f.write(line + '\n')


