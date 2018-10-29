#!/usr/bin/env python

# Returns obj_val function to be used in an optimizer
# A better and updated version of qaoa_obj.py

import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
from networkx.generators.classic import barbell_graph
import copy
import sys
import warnings

import qcommunity.modularity.graphs as gm
from qcommunity.utils.import_graph import generate_graph
from ibmqxbackend.ansatz import IBMQXVarForm


def get_obj(n_nodes,
            B,
            C=None,
            obj_params='ndarray',
            sign=1,
            backend='IBMQX',
            backend_params={'depth': 3},
            return_x=False):
    """
    :param obj_params: defines the signature of obj_val function. 'beta gamma' or 'ndarray' (added to support arbitrary number of steps and scipy.optimize.minimize.) 

    :return: obj_val function, number of variational parameters
    :rtype: tuple
    """
    if return_x:
        all_x = []
        all_vals = []
    # TODO refactor, remove code duplication
    if backend == 'IBMQX':
        var_form = IBMQXVarForm(
            num_qubits=n_nodes, depth=backend_params['depth'])
        num_parameters = var_form.num_parameters
        if obj_params == 'ndarray':

            def obj_val(x):
                resstrs = var_form.run(x)
                modularities = [
                    gm.compute_modularity(n_nodes, B, x, C=C) for x in resstrs
                ]
                y = np.mean(modularities)
                if return_x:
                    all_x.append(copy.deepcopy(x))
                    all_vals.append({'max': max(modularities), 'mean': y})
                print("Actual modularity (to be maximized): {}".format(y))
                return sign * y
        else:
            raise ValueError(
                "obj_params '{}' not compatible with backend '{}'".format(
                    obj_params, backend))
    else:
        raise ValueError("Unsupported backend: {}".format(backend))

    if return_x:
        return obj_val, num_parameters, all_x, all_vals
    else:
        return obj_val, num_parameters


def get_obj_val(graph_generator_name,
                left,
                right,
                seed=None,
                obj_params='ndarray',
                sign=1,
                backend='IBMQX',
                backend_params={'depth': 3},
                return_x=False):
    # Generate the graph
    G, _ = generate_graph(graph_generator_name, left, right, seed=seed)
    B = nx.modularity_matrix(G).A
    return get_obj(
        G.number_of_nodes(),
        B,
        obj_params=obj_params,
        sign=sign,
        backend=backend,
        backend_params=backend_params,
        return_x=return_x)


if __name__ == "__main__":
    x = np.array([2.1578616206475347, 0.1903995547630178])
    obj_val, _ = get_obj_val("get_barbell_graph", 3, 3)
    print(obj_val(x[0], x[1]))
    obj_val, num_parameters = get_obj_val(
        "get_barbell_graph", 3, 3, obj_params='ndarray', backend='IBMQX')
    y = np.random.uniform(-np.pi, np.pi, num_parameters)
    print(obj_val(y))
    obj_val, num_parameters = get_obj_val(
        "get_barbell_graph", 3, 3, obj_params='ndarray')
    z = np.random.uniform(-np.pi, np.pi, num_parameters)
    print(obj_val(z))
