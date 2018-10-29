#!/usr/bin/env python

# solves modularity for large-ish problems using local refinement
# saves run result in a pickle with hardcoded address
#
# Examples:
#
# ./single_level_refinement.py
# ./single_level_refinement.py --method qaoa
# ./single_level_refinement.py -g get_random_partition_graph -l 10 -r 15
# ./single_level_refinement.py --graph /zfs/safrolab/share/dwave_vs_qaoa/graphs/brunson_revolution/out.brunson_revolution_revolution --method brute --iter-size 15 --verbose
# mpirun -np 4 python -m mpi4py single_level_refinement.py --method qaoa --qaoa-method libensemble --mpi --verbose
#
# ./single_level_refinement.py --graph /zfs/safrolab/share/dwave_vs_qaoa/graphs/arenas-jazz/out.arenas-jazz --method optimal --verbose --stopping-criteria 3 --seed 0

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import copy
import random
import argparse
import logging
from operator import itemgetter
from collections import deque
import multiprocessing
from multiprocessing import Pool
import pickle
import time
import progressbar
import qcommunity.modularity.graphs as gm
import qcommunity.modularity.optimal as opt
from qcommunity.utils.import_graph import import_konect, generate_graph, import_pajek, import_edgelist


def bfs_populate_subset(G, root, subset_size):
    if G.number_of_nodes() <= subset_size:
        return list(G.nodes())
    curr_layer = deque(G.neighbors(root))
    res_subset = {root}
    while len(res_subset) < subset_size and curr_layer:
        curr_el = curr_layer.popleft()
        next_layer = set(G.neighbors(curr_el)) - res_subset
        res_subset.add(curr_el)
        curr_layer.extend(next_layer)
    return list(res_subset)


def top_gains_populate_subset(G, subset_size, gains):
    if G.number_of_nodes() <= subset_size:
        return list(G.nodes())
    return [
        x[0] for x in sorted(gains.items(), key=itemgetter(1))[-subset_size:]
    ]


def spectral_populate_subset(G, root, subset_size, gains, threshold):
    if G.number_of_nodes() <= subset_size:
        return list(G.nodes())
    ordering = nx.spectral_ordering(G)
    logging.info("Ordering: {}, root: {}, threshold: {}".format(
        ordering, root, threshold))
    left_it = ordering.index(root) - 1
    right_it = left_it + 2
    res_subset = {root}
    while len(res_subset) < subset_size:
        # move two pointers and add encountered vertices if the gain is larger than threshold
        try:
            left_cand = ordering[left_it]
            if gains[left_cand] >= threshold:
                res_subset.add(left_cand)
                logging.info("Adding ordering[{}]={} with gain {}".format(
                    left_it, left_cand, gains[left_cand]))
            left_it -= 1
        except IndexError:
            pass
        try:
            right_cand = ordering[right_it]
            if gains[right_cand] >= threshold:
                res_subset.add(right_cand)
                logging.info("Adding ordering[{}]={} with gain {}".format(
                    right_it, right_cand, gains[right_cand]))
            right_it += 1
        except IndexError:
            pass
    return list(res_subset)


def iteration_step(G,
                   B,
                   curr_solution,
                   subset,
                   method='brute',
                   method_params=None,
                   qaoa_method='bayes',
                   backend='IBMQX',
                   backend_params={
                       'backend_device': None,
                       'depth': 3
                   }):
    # subset should be a list

    # indices conversion
    subset2global = dict((x, subset[x]) for x in range(0, len(subset)))
    global2subset = dict((subset[x], x) for x in range(0, len(subset)))

    # \Sigma_{i>j\in V_m}B_{ij}s_is_j + \Sigma_{i\in V_m}C_{i}s_i
    # Coefficients for the second part (C_i)
    C = [0.0] * len(subset)
    for i in subset:
        for j in set(G.nodes()) - set(subset):
            C[global2subset[i]] += 2 * B[i, j] * curr_solution[j]
    indices = np.array(subset)  # rows and columns of B to keep for subset
    if method == 'qaoa':
        if backend_params['backend_device'] is None:
            params = {'init_points': 15, 'n_iter': 15}
        else:
            # running on very expensive device, makes sense to spend more time optimizing parameters
            params = {'init_points': 20, 'n_iter': 80}
        _, optimized_subset = qaoa_opt.optimize_modularity(
            len(subset),
            B[np.ix_(indices, indices)],
            C,
            params=params,
            method=qaoa_method,
            backend=backend,
            backend_params=backend_params)
        if qaoa_method == 'libensemble':
            print("rank {} optimized_subset {}".format(
                MPI.COMM_WORLD.Get_rank(), optimized_subset))
            MPI.COMM_WORLD.Barrier()
    elif method == 'brute':
        # good ol' brute force
        _, optimized_subset = gm.optimize_modularity(
            len(subset), B[np.ix_(indices, indices)], C)
    elif method == 'dwave':
        _, optimized_subset = dwave_opt.optimize_modularity(
            len(subset), B[np.ix_(indices, indices)], C,
            method_params['solver'], method_params['embedding'])
    elif method == 'optimal':
        _, optimized_subset = opt.optimize_modularity(
            len(subset), B[np.ix_(indices, indices)], C,
            method_params['timelimit'])
    else:
        raise ValueError("Invalid method {}".format(method))
    if 0 in optimized_subset:
        # assuming optimized_subset is of zeros and ones
        optimized_subset = [
            -1 if x == 0 else 1 if x == 1 else 'Error' for x in optimized_subset
        ]
    if 'Error' in optimized_subset or len(optimized_subset) != len(subset):
        raise ValueError(
            "Incorrect bistring encountered. Only accepts bitstrings containing 0s and 1s or -1s and 1s of the size equal to number of nodes in G. optimized_subset: {}, len(optimized_subset)={}, len(subset)={}"
            .format(optimized_subset, len(optimized_subset), len(subset)))
    for i in range(0, len(optimized_subset)):
        curr_solution[subset2global[i]] = optimized_subset[i]
    return curr_solution


# for MPI allreduce
def opTupleMax(a, b):
    return max(a, b, key=itemgetter(0))


def single_level_optimize_modularity(G,
                                     solution_bitstring=None,
                                     random_seed=42,
                                     size_of_iteration=12,
                                     method='brute',
                                     subset_selection='spectral',
                                     stopping_criteria=3,
                                     method_params=None,
                                     qaoa_method='bayes',
                                     backend='IBMQX',
                                     backend_params={
                                         'backend_device': None,
                                         'depth': 3
                                     }):
    np.random.seed(random_seed)
    random.seed(random_seed)
    B = nx.modularity_matrix(G, nodelist=sorted(G.nodes()), weight='weight')
    if solution_bitstring is not None:
        logging.info("Solution: {}".format(solution_bitstring))

    # random initial guess
    curr_solution = [
        1 - 2 * x
        for x in list(np.random.randint(2, size=(G.number_of_nodes(),)))
    ]
    curr_modularity = gm.compute_modularity(G, B, curr_solution)
    if solution_bitstring is not None:
        optimal_modularity = gm.compute_modularity(G, B, solution_bitstring)
    else:
        optimal_modularity = float('inf')
    print("Initial guess: {}, optimal: {}".format(curr_modularity,
                                                  optimal_modularity))

    # We will keep track of all time guess so when we reset after getting stuck in local optima, we don't lose it
    all_time_best_solution = curr_solution
    all_time_best_modularity = curr_modularity

    visited = set()
    it = 0
    it_stuck = 0
    all_modularities = []
    num_cores = min(multiprocessing.cpu_count(), 16)
    logging.info(
        "Using {} for subset selection, {} for subset optimization".format(
            subset_selection, method))
    while set(G.nodes()) - visited:
        # if using mpi, sync the best solution at the start of each iteration
        if qaoa_method == 'libensemble':
            curr_modularity, curr_solution = MPI.COMM_WORLD.allreduce(
                (all_time_best_modularity, all_time_best_solution),
                op=opTupleMax)
        it += 1
        if it_stuck > stopping_criteria:
            logging.info("Exiting at iteration {}".format(it))
            break

        gains_list = []
        for v in progressbar.progressbar(G.nodes()):
            gains_list.append(gm.compute_gain(G, B, curr_solution, v, True))
        gains = {v: gain for gain, v in gains_list}

        if subset_selection == 'spectral':
            notvisited_gains = {
                v: gain for v, gain in gains.items() if v not in visited
            }
            # if stuck, try selecting random vertex to climb out of local optima
            n, curr_gain = max(notvisited_gains.items(), key=itemgetter(1))
            threshold = np.percentile(
                list(gains.values()),
                25) if 0.75 * G.number_of_nodes() >= size_of_iteration else min(
                    list(gains.values()))
            visited.add(n)
            subset = spectral_populate_subset(G, n, size_of_iteration, gains,
                                              threshold)
            logging.info(
                "Iter {}, looking at vertex {} and its neighbors {}, potential gain {}"
                .format(it, n, subset, curr_gain))
        elif subset_selection == 'bfs':
            n, curr_gain = max(gains.items(), key=itemgetter(1))
            visited.add(n)
            subset = bfs_populate_subset(G, n, size_of_iteration)
            logging.info(
                "Iter {}, looking at vertex {} and its neighbors {}, potential gain {}"
                .format(it, n, subset, curr_gain))
        elif subset_selection == 'top_gain':
            subset = top_gains_populate_subset(G, size_of_iteration, gains)
            logging.info("Iter {}, looking at {}".format(it, subset))
        else:
            raise ValueError(
                "Invalid subset selection method: {}".format(subset_selection))
        logging.info("curr_solution:\t{}".format(curr_solution))
        if logging.getLogger().getEffectiveLevel() >= logging.INFO:
            subproblem = copy.deepcopy(curr_solution)
            for v in subset:
                subproblem[v] = "*"
            logging.info("Subproblem:\t{}\tcurr_modularity\t{}".format(
                subproblem, curr_modularity))
        cand_solution = iteration_step(
            G,
            B,
            copy.deepcopy(curr_solution),
            list(subset),
            method=method,
            method_params=method_params,
            qaoa_method=qaoa_method,
            backend=backend,
            backend_params=backend_params)
        cand_modularity = gm.compute_modularity(G, B, cand_solution)
        logging.info("Solution:\t{}\tcand_modularity\t{}".format(
            cand_solution, cand_modularity))
        print('it', it, 'cand_modularity', cand_modularity, 'curr_best',
              all_time_best_modularity)
        all_modularities.append({
            'it': it,
            'cand_modularity': cand_modularity,
            'curr_best': all_time_best_modularity
        })
        if cand_modularity > curr_modularity:
            curr_solution = cand_solution
            curr_modularity = cand_modularity
            it_stuck = 0
        #    logging.info("New modularity found: {}".format(curr_modularity))
        else:
            it_stuck += 1
        #    logging.info("Ignoring modularity: {}".format(cand_modularity))
        if curr_modularity > all_time_best_modularity:
            all_time_best_solution = curr_solution
            all_time_best_modularity = curr_modularity
        if all_time_best_modularity >= 0.95 * optimal_modularity:
            logging.info(
                "Found really good solution at iter {}, exiting".format(it))
            break
    return (all_time_best_modularity, all_time_best_solution, it,
            all_modularities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        type=int,
        default=15,
        help="number of vtx in the left (first) community")
    parser.add_argument(
        "-r",
        type=float,
        default=17,
        help="number of vtx in the right (second) community")
    parser.add_argument(
        "--iter-size",
        type=int,
        default=12,
        help="size of subproblem in each iteration")
    parser.add_argument(
        "--stopping-criteria",
        type=int,
        default=3,
        help=
        "number of iterations of no improvement after which the refinement is stopped"
    )
    parser.add_argument(
        "--pyomo-timelimit",
        type=float,
        default=100.0,
        help="time limit for pyomo solver")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "-g",
        "--graph-generator",
        type=str,
        default="get_random_partition_graph",
        help="graph generator function")
    parser.add_argument(
        "--label",
        type=str,
        help=
        "description of this version of the script. The description is prepended to the filename, so it should not contain any spaces. Default: time stamp"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='brute',
        choices=['qaoa', 'brute', 'optimal'],
        help="method to be used (brute is brute force)")
    parser.add_argument(
        "--qaoa-method",
        type=str,
        default='COBYLA',
        choices=['neldermead', 'libensemble', 'COBYLA'],
        help="method used internally for qaoa parameter optimization")
    parser.add_argument(
        "--backend",
        type=str,
        default="IBMQX",
        choices=["IBMQX"],
        help="backend simulator to be used")
    parser.add_argument(
        "--backend-device",
        type=str,
        default=None,
        help="backend device name (training on simulator, running on device)")
    parser.add_argument(
        "--computed",
        help="path to pickle with a set of already computed results",
        type=str)
    parser.add_argument(
        "--backend-ansatz-depth",
        type=int,
        default=1,
        help="backend ansatz depth (only for IBMQX)")
    parser.add_argument(
        "--subset",
        type=str,
        default='spectral',
        choices=['spectral', 'top_gain', 'bfs'],
        help=
        "subset (subproblem) selection method (spectral is highest gain and its neighbors in spectral ordering, bfs is highest gain and its neighbors in bfs fashion, top_gain is greedy highest gain)"
    )
    parser.add_argument(
        "--verbose", help="sets logging level to INFO", action="store_true")
    parser.add_argument(
        "--graph",
        type=str,
        help="path to KONECT edgelist (out.graphname file)")
    parser.add_argument(
        "--pajek", type=str, help="path to graph in pajek format")
    parser.add_argument(
        "--edgelist",
        type=str,
        help="path to graph in edgelist format (nx.write_edgelist)")
    parser.add_argument(
        "--mpi",
        help="use this flag if running with mpi (i.e. with libensemble)",
        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.qaoa_method == 'libensemble' and not args.mpi:
        raise ValueError(
            'Have to use --mpi flag when running with libensemble!')

    main_proc = True

    method_params = {
    }  # used to pass extra parameters to subproblem solver, like embedding for dwave. Dictionary.
    backend_params = {
        'backend_device': args.backend_device,
        'depth': args.backend_ansatz_depth
    }

    if args.backend_device is not None and not args.computed:
        logging.warning(
            "Not checking for already computed results for backend device {}"
            .format(args.backend_device))

    # Only supported for KONECT for now
    if args.computed and args.graph:
        computed = pickle.load(open(args.computed, "rb"))
        if "{}{}".format(os.path.basename(args.graph), args.seed) in computed:
            print(
                "Found precomputed result for {} with seed {}, exiting".format(
                    os.path.basename(args.graph), args.seed))
            sys.exit(0)

    if args.method == 'qaoa':
        import qcommunity.optimization.optimize as qaoa_opt
    elif args.method == 'optimal':
        method_params['timelimit'] = args.pyomo_timelimit

    if args.graph:
        # load from edgelist
        G = import_konect(args.graph)
        solution_bitstring = None
        graph_name = os.path.basename(args.graph)
    elif args.pajek:
        G = import_pajek(args.pajek)
        solution_bitstring = None
        graph_name = os.path.basename(args.pajek)
    elif args.edgelist:
        G = import_edgelist(args.edgelist)
        solution_bitstring = None
        graph_name = os.path.basename(args.edgelist)
    else:
        # generate
        G, solution_bitstring = generate_graph(
            args.graph_generator, args.l, args.r, seed=args.seed)
        graph_name = None

    if args.label:
        label = args.label
    else:
        import time
        label = time.strftime("%Y%m%d-%H%M%S")

    if args.graph or args.pajek or args.edgelist:
        outname = "data/out/{}_{}_seed_{}_method_{}_iter_size_{}_backend_{}_pyomo_timelimit_{}".format(
            label, graph_name, args.seed, args.method, args.iter_size,
            args.backend, args.pyomo_timelimit)
    else:
        outname = "data/out/{}_{}_left_{}_right_{}_seed_{}_method_{}_iter_size_{}_backend_{}_pyomo_timelimit_{}".format(
            label, args.graph_generator, args.l, args.r, args.seed, args.method,
            args.iter_size, args.backend, args.pyomo_timelimit)
    print("Output path: ", outname)
    if os.path.isfile(outname):
        print('Output file {} already exists! Better quit before doing anything'
              .format(outname))
        sys.exit(1)

    best_found_modularity, best_found_bitstring, it, all_modularities = single_level_optimize_modularity(
        G,
        solution_bitstring=solution_bitstring,
        random_seed=args.seed,
        size_of_iteration=args.iter_size,
        method=args.method,
        subset_selection=args.subset,
        stopping_criteria=args.stopping_criteria,
        method_params=method_params,
        qaoa_method=args.qaoa_method,
        backend=args.backend,
        backend_params=backend_params)
    if solution_bitstring is not None:
        optimal_modularity = gm.compute_modularity_c(G, solution_bitstring)
    else:
        optimal_modularity = None
    print("\n\nFound modularity {} after {} iterations, optimal {}".format(
        best_found_modularity / (4.0 * G.number_of_edges()), it,
        optimal_modularity))

    if main_proc:
        res = {
            'optimal_modularity':
                optimal_modularity,
            'best_found_modularity':
                best_found_modularity,
            'best_found_modularity_scaled': (
                best_found_modularity / (4.0 * G.number_of_edges())),
            'best_found_bitstring':
                best_found_bitstring,
            'graph':
                graph_name,
            'l':
                args.l,
            'r':
                args.r,
            'graph_generator':
                args.graph_generator,
            'seed':
                args.seed,
            'label':
                label,
            'n_iter':
                it,
            'all_modularities':
                all_modularities,
            'backend':
                args.backend,
            'backend_params':
                backend_params,
            'method_params':
                method_params,
            'iter_size':
                args.iter_size,
            'args':
                args
        }
        pickle.dump(res, open(outname, "wb"))
        print("Dumped pickle to ", outname)
