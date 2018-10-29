#!/usr/bin/env python

# Builds gap figure for the PRL paper
#
# With correct seed:
#
# ./both_figures.py --gap data/random_modular_random_modular_graph_2000_12_2_q_0.45.p_seed_1_method_optimal_iter_size_200_backend_IntelQS_pyomo_timelimit_* --subprob {final_prl_more_itersizes_correct_seed_random_modular_graph_2000_12_2_q_0.45.p_seed_1_method_optimal_iter_size_*1000.0,data/random_modular_random_modular_graph_2000_12_2_q_0.45.p_seed_1_method_optimal_iter_size_200_backend_IntelQS_pyomo_timelimit_1000.0}

import argparse
import sys
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import os.path
from operator import itemgetter
import csv
import re
from qcommunity.utils.import_graph import import_konect


def get_first_int_after_substring(output, str_to_look_for):
    outsplit = output.split(str_to_look_for)
    if len(outsplit) < 2:
        return None
    outclean = []
    for s in outsplit:
        outclean.append(s.strip())

    return int(float((re.match('\d+', outclean[-1])).group(0)))


def get_first_float_after_substring(output, str_to_look_for):
    outsplit = output.split(str_to_look_for)
    if len(outsplit) < 2:
        return None
    outclean = []
    for s in outsplit:
        outclean.append(s.strip())

    return float((re.match('\d+\.\d+', outclean[-1])).group(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gap",
        type=str,
        nargs='+',
        help="path to saved results for the gap plot (left)")
    parser.add_argument(
        "--subprob",
        type=str,
        nargs='+',
        help=
        "path to saved results for the plot with varying subproblem size (right)"
    )
    parser.add_argument(
        "--csv", help="path to csv file to put results", type=str)
    parser.add_argument(
        "--graph",
        type=str,
        help="path to KONECT edgelist (out.graphname file)")
    args = parser.parse_args()

    if args.graph:
        G = import_konect(args.graph)
        scaling_factor = 4.0 * G.number_of_edges()
    else:
        # No scaling
        scaling_factor = 1.0

    #colors = ['blue', 'orange', 'aquamarine', 'brown']
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    colors_subpr = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    mpl.style.use('seaborn-colorblind')

    all_res_gap = []
    all_modularities_gap = {}
    for fname in args.gap:
        res = pickle.load(open(fname, "rb"))
        scaling_factor = res['best_found_modularity'] / res[
            'best_found_modularity_scaled']
        all_res_gap.append([
            os.path.basename(fname), res['seed'], res['best_found_modularity'],
            res['best_found_modularity'] / scaling_factor, res['n_iter'],
            res['graph']
        ])
        all_modularities_gap[os.path.basename(fname)] = [
            x['curr_best'] / scaling_factor for x in res['all_modularities']
        ]

    for res in all_res_gap:
        print(res)

    all_res_subprob = []
    all_modularities_subprob = {}
    for fname in args.subprob:
        res = pickle.load(open(fname, "rb"))
        scaling_factor = res['best_found_modularity'] / res[
            'best_found_modularity_scaled']
        all_res_subprob.append([
            os.path.basename(fname), res['seed'], res['best_found_modularity'],
            res['best_found_modularity'] / scaling_factor, res['n_iter'],
            res['graph']
        ])
        all_modularities_subprob[os.path.basename(fname)] = [
            x['curr_best'] / scaling_factor for x in res['all_modularities']
        ]

    for res in all_res_subprob:
        print(res)

    lines = []
    labels = ['Gurobi', 'Quantum (projected)']
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    for i, kv in enumerate(all_modularities_gap.items()):
        k, v = kv
        line, = ax1.plot(v, color=colors[i])
        lines.append(line)
#        labels.append("Gurobi time limit: {}".format(get_first_float_after_substring(k, 'pyomo_timelimit_')))
    ax1.legend(lines, labels)
    ax1.xaxis.set_ticks(
        np.arange(
            0, max([len(x) for k, x in all_modularities_gap.items()]), step=2))

    lines = []
    labels = []
    subproblem_sizes = []
    for i, kv in enumerate(all_modularities_subprob.items()):
        k, v = kv
        if 'random_modular_random_modular_graph_2000_12_2_q_0.45' in k:
            line, = ax2.plot(v, color=colors[1])
        else:
            line, = ax2.plot(v, color=colors_subpr[i])
        lines.append(line)
        labels.append("Subproblem size: {}".format(
            get_first_int_after_substring(k, 'iter_size_')))
        subproblem_sizes.append(get_first_int_after_substring(k, 'iter_size_'))

    lines, labels, _ = zip(
        *sorted(zip(lines, labels, subproblem_sizes),
                key=itemgetter(2)))  # zip, sort and unzip back
    ax2.legend(lines, labels)
    ax2.xaxis.set_ticks(
        np.arange(0, max([len(x) for x in all_modularities_subprob]), step=10))

    f.subplots_adjust(wspace=0.0)
    f.subplots_adjust(left=0.05)
    f.subplots_adjust(right=0.98)
    outfname = "gap.pdf"
    plt.savefig(outfname, dpi=300, format='pdf')

    if args.csv:
        with open(args.csv, 'w') as f:
            w = csv.writer(f)
            for r in all_res_gap:
                w.writerow(r)
