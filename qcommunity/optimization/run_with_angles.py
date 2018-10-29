#!/usr/bin/env python

# Tests the angles produced by optimization routine

# usage: ./test_angles.py -g get_random_partition_graph -l 6 -r 7

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from networkx.generators.classic import barbell_graph
from itertools import product
import sys
import argparse
import random
import pickle
from operator import itemgetter

import qcommunity.modularity.graphs as gm
from qcommunity.utils.import_graph import generate_graph
from ibmqxbackend.ansatz import IBMQXVarForm


def run_angles(n_nodes,
               B,
               angles,
               C=None,
               backend='IBMQX',
               backend_params={
                   'backend_device': None,
                   'depth': 3
               }):
    if backend == 'IBMQX':
        if not isinstance(angles, (np.ndarray, np.generic, list)):
            raise ValueError(
                "Incorrect angles received: {} for backend {}".format(
                    angles, backend))
        var_form = IBMQXVarForm(
            num_qubits=n_nodes, depth=backend_params['depth'])
        resstrs = var_form.run(
            angles, backend_name=backend_params['backend_device'])
    else:
        raise ValueError("Unsupported backend: {}".format(backend))
    modularities = [
        (gm.compute_modularity(n_nodes, B, x, C=C), x) for x in resstrs
    ]
    return max(modularities, key=itemgetter(0))


def test_angles(graph_generator_name,
                left,
                right,
                angles,
                seed=None,
                verbose=0,
                compute_optimal=False,
                backend='IBMQX',
                backend_params={
                    'backend_device': None,
                    'depth': 3
                }):
    # note that compute optimal uses brute force! Not recommended for medium and large problem
    # angles should be a dictionary with fields 'beta' and 'gamma', e.g. {'beta': 2.0541782343349086, 'gamma': 0.34703642333837853}

    rand_seed = seed

    # Generate the graph
    G, _ = generate_graph(graph_generator_name, left, right, seed=seed)
    # Use angles

    # Using NetworkX modularity matrix
    B = nx.modularity_matrix(G).A

    # Compute ideal cost
    if compute_optimal:
        optimal_modularity = gm.compute_modularity(G, B, solution_bitstring)
        print("Optimal solution energy: ", optimal_modularity)
    else:
        optimal_modularity = None

    if backend == 'IBMQX':
        if not isinstance(angles, (np.ndarray, np.generic, list)):
            raise ValueError(
                "Incorrect angles received: {} for backend {}".format(
                    angles, backend))
        var_form = IBMQXVarForm(
            num_qubits=G.number_of_nodes(), depth=backend_params['depth'])
        resstrs = var_form.run(angles)
    else:
        raise ValueError("Unsupported backend: {}".format(backend))

    if verbose > 1:
        # print distribution
        allstrs = list(product([0, 1], repeat=len(qubits)))
        freq = {}
        for bitstr in allstrs:
            freq[str(list(bitstr))] = 0
        for resstr in resstrs:
            resstr = str(list(resstr))  # for it to be hashable
            if resstr in freq.keys():
                freq[resstr] += 1
            else:
                raise ValueError("received incorrect string: {}".format(resstr))
        for k, v in freq.items():
            print("{} : {}".format(k, v))

    # Raw results
    modularities = [gm.compute_modularity(G, B, x) for x in resstrs]
    mod_max = max(modularities)
    # Probability of getting best modularity
    if compute_optimal:
        mod_pmax = float(np.sum(np.isclose(
            modularities, optimal_modularity))) / float(len(modularities))
    else:
        mod_pmax = None
    mod_mean = np.mean(modularities)
    if verbose:
        print("Best modularity found:", mod_max)
        print("pmax: ", mod_pmax)
        print("mean: ", mod_mean)
    return {
        'max': mod_max,
        'mean': mod_mean,
        'pmax': mod_pmax,
        'optimal': optimal_modularity,
        'x': angles
    }
