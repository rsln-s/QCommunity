#!/usr/bin/env python
# QAOA parameter optimization using Nedler-Mead

from qcommunity.optimization.obj import get_obj_val, get_obj
from scipy.optimize import minimize
import numpy as np


def optimize_obj(obj_val, num_parameters, params=None):
    options = {}
    try:
        init_points = params['initial_guess']
    except (KeyError, TypeError):
        init_points = np.random.uniform(-np.pi, np.pi, num_parameters)
    try:
        options['maxfev'] = params['n_iter'] + params['init_points']
    except (KeyError, TypeError):
        options['maxfev'] = 100
    options['return_all'] = True
    res = minimize(obj_val, init_points, method='Nelder-Mead', options=options)
    return res
