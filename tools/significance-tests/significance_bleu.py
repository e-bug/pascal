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

import sacrebleu
import numpy as np
import multiprocessing


def get_parser():
    parser = argparse.ArgumentParser()

    # Translation files
    parser.add_argument('reference_dir', type=str, help='path to reference translations file')
    parser.add_argument('system_x_dir', type=str, help='path to system X translations file')
    parser.add_argument('system_y_dir', type=str, help='path to system Y translations file')

    # General options
    parser.add_argument('--seed', type=int, default=1234, help='pseudo random number generator seed')
    parser.add_argument('--p-vals', type=float, nargs='+', default=[0.01, 0.05], help='number of sentences per sample')
    parser.add_argument('--num-processes', type=int, default=None, help='number of worker processes')
    parser.add_argument('--tok', type=str, default='13a', help='tokenizer in SacreBLEU')

    # Significance tests options
    parser.add_argument('--test-method', default='bootstrap_resampling', help='randomized significance test method',
                        choices=['bootstrap_resampling', 'paired_bootstrap_resampling'])
    parser.add_argument('--two-sided-bootstrap', type=bool, default=False, help='use two-sided bootstrap resampling')
    parser.add_argument('--num-bootstrap-samples', '-B', type=int, default=1000, help='number of bootstreap samples')
    parser.add_argument('--sample-size', '-S', type=int, default=None, help='number of sentences per sample')

    return parser


def pseudo_scores(ixs):
    return np.array(score_fn(x_translations[ixs], ref_translations[ixs])), \
           np.array(score_fn(y_translations[ixs], ref_translations[ixs]))


def bootstrap_resampling_test(num_samples, sample_size, actual_diff, two_sided=False, **kwargs):
    # bootstrap sample indices
    sample_ixs = [np.random.choice(range(len(ref_translations)), sample_size) for _ in range(num_samples)]

    # calculate pseudo score differences
    sample_scores = pool.map(pseudo_scores, sample_ixs)
    sample_diffs = list(map(lambda t: t[0] - t[1], sample_scores))

    # calculate sample mean
    sample_mean = np.sum(sample_diffs) / num_samples

    # compute pseudo-difference on bootstrap data
    if two_sided:
        actual_diff = np.abs(actual_diff)
        return np.sum(np.abs(diff - sample_mean) > actual_diff for diff in sample_diffs) / num_samples
    return np.sum(diff - sample_mean > actual_diff for diff in sample_diffs) / num_samples


def paired_bootstrap_resampling(num_samples, sample_size, **kwargs):
    # bootstrap sample indices
    sample_ixs = [np.random.choice(range(len(ref_translations)), sample_size) for _ in range(num_samples)]

    # compute pseudo comparisons
    sample_scores = pool.map(pseudo_scores, sample_ixs)
    sample_comparisons = list(map(lambda t: t[0] < t[1], sample_scores))

    # compute pseudo-comparison on bootstrap data
    return np.sum(sample_comparisons) / num_samples


def significance_test(test_method, two_sided, num_samples, sample_size, actual_diff):
    if test_method == 'bootstrap_resampling':
        return bootstrap_resampling_test(num_samples, sample_size, actual_diff, two_sided=two_sided)
    elif test_method == 'paired_bootstrap_resampling':
        return paired_bootstrap_resampling(num_samples, sample_size)
    return approximate_randomization_test()


if __name__ == '__main__':

    # Init
    global ref_translations, x_translations, y_translations
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    num_processes = multiprocessing.cpu_count() if args.num_processes is None else args.num_processes
    score_fn = lambda preds, refs: sacrebleu.corpus_bleu(preds, [refs], tokenize=args.tok).score

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

    # overall scores
    x_score = score_fn(x_translations, ref_translations)
    y_score = score_fn(y_translations, ref_translations)
    print('System X: %f' % x_score)
    print('System Y: %f' % y_score)

    # actual score differences
    actual_diff = x_score - y_score
    print('Actual score difference: %f' % actual_diff)

    # evaluate probability of observing a more extreme value
    pool = multiprocessing.Pool(num_processes)
    extreme_val_probab = significance_test(args.test_method, args.two_sided_bootstrap,
                                           args.num_bootstrap_samples, sample_size, actual_diff)
    pool.close()

    # reject null hypothesis if probability of observing a more extreme value
    # than the actual statistic is lower than p_val
    for p_val in sorted(args.p_vals):
        if extreme_val_probab <= p_val:
            print('System X is significantly better at level %.2f (p < %.2f)' % (1 - p_val, p_val))
            exit(0)
    print('No significant difference in scores: %.2f' % extreme_val_probab)

