#!/usr/bin/env python

import os
import tempfile
import multiprocessing
import minimize_ising_model
from pyomo.environ import *


def ising_to_file(B_matrix, bias, fname):
    data_var = {}
    data_var['couplers'] = 'set couplers :=\n'
    data_var['nodes'] = 'set nodes :=\n'
    data_var['bias'] = 'param bias := \n'
    data_var['weight'] = 'param w := \n'

    mygraphfile = open(fname, 'w')
    n = B_matrix.shape[0]
    # Take negative values because we max modularity
    B_matrix = -B_matrix
    bias = [-i for i in bias]

    for i in range(n - 1):
        for j in range(i, n):
            w = B_matrix.item((i, j))
            data_var['couplers'] += ' '.join([str(i), str(j), '\n'])
            data_var['weight'] += ' '.join([str(i), str(j), str(w), '\n'])
    i, j = n - 1, n - 1
    w = B_matrix.item((i, j))
    data_var['couplers'] += ' '.join([str(i), str(j), '\n'])
    data_var['weight'] += ' '.join([str(i), str(j), str(w), '\n'])

    for i in range(n):
        data_var['nodes'] += str(i) + '\n'
        data_var['bias'] += str(i) + ' ' + str(bias[i]) + '\n'

    data_var['nodes'] += ';\n'
    data_var['bias'] += ';\n'
    data_var['weight'] += ';\n'
    data_var['couplers'] += ';\n'

    for item in data_var:
        mygraphfile.write(data_var[item])


def pyomo_solve(sub_B_matrix, bias, time_limit=100):
    with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.dat', dir=os.environ['TMPDIR']) as modelfile:
        ising_to_file(sub_B_matrix, bias, modelfile.name)
        instance = minimize_ising_model.model.create_instance(modelfile.name)
    solver = SolverFactory("gurobi")
    solver.options['mipgap'] = 0.00000001
    solver.options['threads'] = min(16, multiprocessing.cpu_count())
    solver.options['timelimit'] = time_limit
    results = solver.solve(
        instance, tee=True)  # tee=True prints gurobi solve info
    energy = instance.min_ising()
    #print energy, constant, energy - constant
    # Get partition
    varobject = getattr(instance, 'x')
    part0 = []
    part1 = []
    n = sub_B_matrix.shape[0]
    ising_partition = ['unset' for i in range(n)]
    for index in sorted(varobject):
        x_val = varobject[index].value
        s_val = 2 * x_val - 1
        if s_val < 0:
            s_val = -1
        else:
            s_val = 1
        assert s_val != 0
        ising_partition[index] = s_val

    return energy, ising_partition


def optimize_modularity(n_nodes, B, C, time_limit=100):
    return pyomo_solve(B, C, time_limit=time_limit)
