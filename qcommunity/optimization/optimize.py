#!/usr/bin/env python

# QAOA parameter optimization

# Example: mpirun -np 2 python -m mpi4py optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method libensemble --mpi --backend IBMQX
# Example: ./optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method neldermead
# Example: ./optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method COBYLA --niter 100 --backend IBMQX

import pickle
import numpy as np
import os.path
import sys
import argparse
import warnings
import random
from operator import itemgetter
from qcommunity.optimization.obj import get_obj_val, get_obj
from qcommunity.optimization.run_with_angles import run_angles, test_angles
import qcommunity.optimization.neldermead as nm
import qcommunity.optimization.cobyla as cobyla


def optimize_modularity(n_nodes,
                        B,
                        C=None,
                        params=None,
                        method='COBYLA',
                        backend='IBMQX',
                        backend_params={
                            'backend_device': None,
                            'depth': 3
                        }):
    if method == 'neldermead':
        obj_val, num_parameters = get_obj(
            n_nodes,
            B,
            C,
            obj_params='ndarray',
            sign=-1,
            backend=backend,
            backend_params=backend_params
        )  # sign = -1 because neldermead minimizes
        res = nm.optimize_obj(obj_val, num_parameters, params)
        optimized = run_angles(
            n_nodes,
            B,
            res.x,
            C=C,
            backend=backend,
            backend_params=backend_params)
    elif method == 'COBYLA':
        obj_val, num_parameters = get_obj(
            n_nodes,
            B,
            C,
            obj_params='ndarray',
            sign=-1,
            backend=backend,
            backend_params=backend_params)  # sign = -1 because COBYLA minimizes
        res = cobyla.optimize_obj(obj_val, num_parameters, params)
        optimized = run_angles(
            n_nodes,
            B,
            res.x,
            C=C,
            backend=backend,
            backend_params=backend_params)
    else:
        raise ValueError('Incorrect method: {}'.format(method))
    return optimized
