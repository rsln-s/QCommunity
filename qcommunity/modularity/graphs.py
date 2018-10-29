#!/usr/bin/env python

# Graphs and bistrings with optimal modularity solutions

import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators.classic import barbell_graph
from networkx.algorithms.community.community_generators import LFR_benchmark_graph
import warnings
from itertools import product
from operator import itemgetter
import math
import copy
import multiprocessing
from multiprocessing import Pool
import numpy as np


# added to support python2, use with p.map
def _test_all_with_prefix_tuple(args):
    return _test_all_with_prefix(args[0], args[1], args[2], args[3])


def _test_all_with_prefix(prefix, n_nodes, B, C):
    suffix_len = n_nodes - len(prefix)
    curr_best_str = [0] * n_nodes
    curr_best = compute_modularity(n_nodes, B, curr_best_str, C)
    for x in product([0, 1], repeat=suffix_len):
        s = prefix + list(x)
        curr = compute_modularity(n_nodes, B, s, C)
        if curr > curr_best:
            curr_best = curr
            curr_best_str = s
    return (curr_best, curr_best_str)


def optimize_modularity(n_nodes, B, C=None):
    if isinstance(n_nodes, nx.Graph) or isinstance(n_nodes, nx.DiGraph):
        # legacy
        n_nodes = n_nodes.number_of_nodes()
    num_cores = min(multiprocessing.cpu_count(), 16)
    prefix_len = int(math.log(num_cores, 2))
    prefixes = list(product([0, 1], repeat=prefix_len))
    p = Pool(num_cores)
    params = [(list(x), n_nodes, B, C) for x in prefixes]
    results = p.map(_test_all_with_prefix_tuple, params)
    p.close()
    return max(results, key=itemgetter(0))


def get_optimal_modularity_bitstring(G):
    # Guaranteed to return a string of +1 / -1
    B = nx.modularity_matrix(G).A
    _, bitstring = optimize_modularity(G, B)
    if 0 in bitstring:
        # assuming bitstring is of zeros and ones
        bitstring = [
            -1 if x == 0 else 1 if x == 1 else 'Error' for x in bitstring
        ]
    if 'Error' in bitstring or len(bitstring) != G.number_of_nodes():
        raise ValueError("Incorrect bistring returned by nx.erdos_renyi_graph")
    return bitstring


# added to support python2, use with p.map
def compute_gain_tuple(args):
    return compute_gain(args[0], args[1], args[2], args[3], args[4])


def compute_gain(G, B, curr_bitstring, v, return_v=False):
    """
    Computes gain in modularity from reassigning v to a different community, given the current assignment curr_bitstring
    For now, this is done stupidly
    return_v parameter to support nice parallelisation with multiprocessing.Pool
    """
    reassigned = copy.deepcopy(curr_bitstring)
    reassigned[v] = -curr_bitstring[v]
    gain = (
        compute_modularity(G.number_of_nodes(), B, reassigned) -
        compute_modularity(G.number_of_nodes(), B, curr_bitstring))
    if return_v:
        return gain, v
    else:
        return gain


def compute_modularity_c(G, bitstring):
    B = nx.modularity_matrix(G).A
    return compute_modularity(G, B, bitstring)


def compute_modularity(n_nodes, B, bitstring, C=None):
    if isinstance(n_nodes, nx.Graph) or isinstance(n_nodes, nx.DiGraph):
        # legacy
        n_nodes = n_nodes.number_of_nodes()
    if 0 in bitstring:
        # assuming bitstring is of zeros and ones
        bitstring = [
            -1 if x == 0 else 1 if x == 1 else 'Error' for x in bitstring
        ]
    if 'Error' in bitstring or len(bitstring) != n_nodes:
        raise ValueError(
            "Incorrect bistring encountered. Only accepts bitstrings containing 0s and 1s or -1s and 1s of the size equal to number of nodes in G"
        )
    if not isinstance(bitstring, np.ndarray):
        bitstring = np.asarray(bitstring)
    if not isinstance(C, np.ndarray) and C is not None:
        C = np.asarray(C)
    cost = (bitstring.dot(B)).dot(bitstring.T)
    if C is not None:
        cost += C.T.dot(bitstring)
    return float(cost)


def get_simple_graph():
    G = nx.Graph()

    G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)])

    #G.add_edges_from([(0,1),(1,2),(0,2),(2,3),(3,4),(4,5),(3,5),(4,6),(3,6),(5,6)])
    solution_bitstring = get_optimal_modularity_bitstring(G)
    return G, solution_bitstring


def get_barbell_graph(left, right, **kwargs):
    # kwargs are ignored
    if left != right:
        raise ValueError(
            "Barbell graph requires both communities to be of the same size!")
    size = left
    print('Generating {}x{} barbell graph'.format(size, size))
    G = barbell_graph(size, 0)
    solution_bitstring = [-1] * size + [1] * size
    return G, solution_bitstring


def get_qpu_barbell_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 5), (0, 5), (5, 6), (6, 7), (7, 10), (6, 10)])
    solution_bitstring = [-1, -1, -1, 1, 1, 1]
    return G, solution_bitstring


def get_connected_caveman_graph(left, right, **kwargs):
    right = int(right)
    # kwargs are ignored
    if left != right:
        # Assuming left is number of cliques, right is size of each clique
        ncliques = left
        clique_size = right
    else:
        ncliques = 2
        clique_size = left
    print('Generating connected graph with {} cliques of size {} each'.format(
        ncliques, clique_size))
    G = nx.connected_caveman_graph(ncliques, clique_size)
    if ncliques % 2 == 0:
        size_of_part = int(G.number_of_nodes() / 2)
        solution_bitstring = [-1] * size_of_part + [1] * size_of_part
    else:
        size_of_part = int((clique_size * ncliques - 1) / 2)
        solution_bitstring = [-1] * int(
            (clique_size * (ncliques + 1) / 2)) + [1] * int(
                (clique_size * (ncliques - 1) / 2))
    return G, solution_bitstring


def get_random_partition_graph(left,
                               right,
                               p_in=0.9,
                               p_out=0.2,
                               seed=42,
                               **kwargs):
    right = int(right)
    # kwargs are ignored
    print(
        'Generating random partition graph with first partition of size {} and second of size {}'
        .format(left, right))
    sizes = [left, right]
    G = nx.random_partition_graph(sizes, p_in, p_out, seed=seed)
    solution_bitstring = [-1] * sizes[0] + [1] * sizes[1]
    return G, solution_bitstring


def get_erdos_renyi_graph(left, right, p=0.7, seed=42, **kwargs):
    right = int(right)
    # kwargs are ignored
    if left != right:
        warnings.warn("Ignoring right parameter")
    print("Generating Erdos-Renyi graph on {} vertices with p={} and seed={}"
          .format(left, p, seed))
    G = nx.erdos_renyi_graph(left, p, seed=seed)
    B = nx.modularity_matrix(G).A
    return G, None


def get_florentine_families_graph(left, right, **kwargs):
    right = int(right)
    # kwargs are ignored
    G = nx.convert_node_labels_to_integers(nx.florentine_families_graph())
    return G, None


def get_random_regular_graph(left, right, seed=None, **kwargs):
    right = int(right)
    G = nx.random_regular_graph(left, right, seed=seed)
    return G, None


def get_random_regular_graph_weighted(left,
                                      right,
                                      seed=None,
                                      weighting_scheme='power_law',
                                      **kwargs):
    right = int(right)
    G = nx.random_regular_graph(left, right, seed=seed)
    np.random.seed(seed)
    if weighting_scheme == 'power_law':
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = np.random.power(2.5)

    return G, None


def get_lfr_benchmark_graph(left,
                            right,
                            seed=None,
                            average_degree=20,
                            tau1=2,
                            tau2=1.1,
                            **kwargs):
    # kwargs are ignored
    # parameters taken from https://www.nature.com/articles/srep30750/tables/1
    print(
        "right={} is interpreted as mu (the mixing coefficient)".format(right))
    n = left
    mu = right
    minimum_degree = 5
    maximum_degree = 50
    maximum_community = 100
    minimum_community = 10
    G = LFR_benchmark_graph(
        n,
        tau1,
        tau2,
        mu,
        min_degree=minimum_degree,
        max_degree=maximum_degree,
        max_community=maximum_community,
        min_community=minimum_community)
    return G, None


def get_connected_watts_strogatz_graph(n, k, p=0.7, seed=42, **kwargs):
    k = int(k)
    # kwargs are ignored
    G = nx.connected_watts_strogatz_graph(n, k, p, seed=42)
    return G, None


def get_random_bipartite_graph(left, right, seed=42):
    from networkx.algorithms import bipartite
    #aseq = list(int(3*x) for x in (np.random.pareto(1,left)+1) if x < (left / 9) )
    #G = bipartite.preferential_attachment_graph(aseq, 0.8, create_using=nx.Graph)
    G = bipartite.random_graph(int(left), int(right), 0.7, seed=seed)
    return G, None


if __name__ == "__main__":
    G, solution_bitstring = get_random_partition_graph(
        20, 23, p_in=0.9, p_out=0.1, seed=42)
    #G, solution_bitstring = get_connected_caveman_graph(4,30)
    #G, solution_bitstring = get_barbell_graph(5,5)
    #G, solution_bitstring = get_erdos_renyi_graph(10,10)
    #G, solution_bitstring = get_connected_watts_strogatz_graph(15,5)
    print(compute_modularity_c(G, solution_bitstring))
    colors = [
        'r' if x == -1 else 'b' if x == 1 else 'Error'
        for x in solution_bitstring
    ]
    for i in range(4):
        colors[i] = 'yellow'
    label = "Random Partition Graph"
    nx.draw(G, node_color=colors, label=label)
    import os.path
    #outfname = os.path.join(label, '.pdf')
    import re
    outfname = re.sub('[^A-Za-z0-9]+', '', label) + '.pdf'
    print("Saving pdf to {}".format(outfname))
    plt.savefig(outfname, dpi=300, format='pdf')
    #plt.show()
