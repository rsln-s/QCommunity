#!/usr/bin/env python

# solves modularity for large-ish problems using local refinement

# Example:
# ./multiscale.py --graph /zfs/safrolab/users/rshaydu/quantum/data/graphs/subelj_cora/out.subelj_cora_cora
# ./multiscale.py --graph /zfs/safrolab/users/hushiji/graphs/ego-facebook/out.ego-facebook
"""
    Plan:

    1. Build coarsening hierarchy
    2. Salvage as much code as possible from single_level_refinement, with previous level as initial guess (should be fairly straightforward)

"""

import networkx as nx
import logging
import argparse
import qcommunity.modularity.graphs as gm
from qcommunity.modularity.single_level_refinement import spectral_populate_subset, iteration_step
from qcommunity.utils.import_graph import import_konect, generate_graph
from pycoarsen.coarsen import coarsen


def cluster(G):
    """
    Splits graph G into two communities
    :param G: NetworkX graph to cluster
    :return: solution bitstring
    :rtype: list
    """

    logging.info("Computing coarsening hierarchy...")
    hierarchy = coarsen(G)
    logging.info("Done computing coarsening hierarchy")
    curr_solution = None
    for level, (graph, matching) in zip(
            range(len(hierarchy) - 1, -1, -1), reversed(hierarchy)
    ):  # http://christophe-simonis-at-tiny.blogspot.com/2008/08/python-reverse-enumerate.html
        logging.info("Now at level {}".format(level))
        B = nx.modularity_matrix(graph)
        print(graph.number_of_nodes(), B)
        if curr_solution is None:
            # first level -- initial solution
            _, curr_solution = gm.optimize_modularity(graph.number_of_nodes(),
                                                      B)

        print(curr_solution)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graph-generator",
        type=str,
        default="get_random_partition_graph",
        help="graph generator function")
    parser.add_argument(
        "-l",
        type=int,
        default=15,
        help="number of vtx in the left (first) community")
    parser.add_argument(
        "-r",
        type=int,
        default=17,
        help="number of vtx in the right (second) community")
    parser.add_argument(
        "--graph",
        type=str,
        help="path to KONECT edgelist (out.graphname file)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.graph:
        G = import_konect(args.graph)
    else:
        G, _ = generate_graph(
            args.graph_generator, args.l, args.r, seed=args.seed)
    cluster(G)
