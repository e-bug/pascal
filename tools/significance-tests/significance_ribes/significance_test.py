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

    # General options
    parser.add_argument('--seed', type=int, default=1234, help='pseudo random number generator seed')
    parser.add_argument('--p-vals', type=float, nargs='+', default=[0.01, 0.05], help='number of sentences per sample')

    # Significance tests options
    parser.add_argument('--num-bootstrap-samples', '-B', type=int, default=1000, help='number of bootstreap samples')
    parser.add_argument('--sample-size', '-S', type=int, default=None, help='number of sentences per sample')

    return parser


if __name__ == '__main__':

    # Init
    parser = get_parser()
    args = parser.parse_args()

    xs, ys = [], []
    actual_diff = 0
    with open('ribes.x') as f:
        line = f.readlines()[0]
    score = float(line.split()[0])
    print('System X: %f' % score)

    actual_diff += score
    with open('ribes.y') as f:
        line = f.readlines()[0]
    score = float(line.split()[0])
    print('System Y: %f' % score)
    
    actual_diff -= score
    print('Actual score difference: %f' % actual_diff)

    xys = []
    num_samples = args.num_bootstrap_samples
    for i in range(num_samples):
        xy = []
        with open('ribes.x.'+str(i), 'r') as f:
            line = f.readlines()[0]
        score = float(line.split()[0])
        xy.append(score)
        with open('ribes.y.'+str(i), 'r') as f:
            line = f.readlines()[0]
        score = float(line.split()[0])
        xy.append(score)
        xys.append(xy)

    sample_diffs = list(map(lambda t: t[0] - t[1], xys))
    sample_mean = np.sum(sample_diffs) / num_samples
    extreme_val_probab = np.sum([diff - sample_mean > actual_diff for diff in sample_diffs]) / num_samples
    for p_val in sorted(args.p_vals):
        if extreme_val_probab <= p_val:
            print('System X is significantly better at level %.2f (p < %.2f)' % (1 - p_val, p_val))
            exit(0)
    print('No significant difference in scores: %.2f' % extreme_val_probab)

