#!/usr/bin/env python

import re, os, sys
import numpy as np
import networkx as nx
from numpy import linalg as la
from networkx.generators.atlas import *
import random, copy
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from dwave_sapi2.local import local_connection
from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.core import solve_ising
from dwave_sapi2.embedding import find_embedding
from dwave_sapi2.util import get_chimera_adjacency, get_hardware_adjacency
from dwave_sapi2.embedding import embed_problem, unembed_answer
from chimera_embedding import processor
from collections import Counter
import time as mytime
import math
from pyomo.environ import *
sys.path.insert(0, '../Optimal/')
import minimize_ising_model
from networkx.utils import reverse_cuthill_mckee_ordering

#from networkx.algorithms.community.quality import modularity


# native embedding
def get_native_embedding(solver):
    params = solver.properties.get('parameters', {})
    #M = params.get('M', 12)
    #N = params.get('N', 12)
    M = params.get('M', 16)
    N = params.get('N', 16)
    L = params.get('L', 4)
    hardware_adj = get_hardware_adjacency(solver)
    embedder = processor(hardware_adj, M=M, N=N, L=L)
    embedding = embedder.largestNativeClique()
    return embedding


def get_sub_mod_matrix(graph, ptn_variables):
    B_matrix = nx.modularity_matrix(graph, nodelist=sorted(graph.nodes()))
    free_var = []
    fixed_var = []
    fixed_vec = []
    free_nodes = []
    fixed_nodes = []
    for node in sorted(ptn_variables):
        if ptn_variables[node] == 'free':
            free_var.append(True)
            fixed_var.append(False)
            free_nodes.append(node)
        else:
            free_var.append(False)
            fixed_var.append(True)
            fixed_vec.append(ptn_variables[node])
            fixed_nodes.append(node)
    free_var = np.array(free_var)
    sub_B_matrix = B_matrix[free_var][:, free_var]
    bias = []
    for node_i in free_nodes:
        bias_i = 0
        for node_j in fixed_nodes:
            s_j = ptn_variables[node_j]
            bias_i += 2 * s_j * B_matrix.item((node_i, node_j))
        bias.append(bias_i)
    fixed_var = np.array(fixed_var)
    C = B_matrix[fixed_var][:, fixed_var]
    n = C.shape[0]
    vec = np.array(fixed_vec).reshape(n, 1)
    constant = vec.transpose() * C * vec
    return sub_B_matrix, bias, constant.item(0)


def get_new_partition(best_soln, ptn_variables):
    ''' map sub problem solution to original problem '''
    partition = []
    for node in sorted(ptn_variables):
        if ptn_variables[node] == 'free':
            ptn = best_soln[0]
            partition.append(ptn)
            del best_soln[0]

        else:
            partition.append(ptn_variables[node])
    assert len(partition) == len(ptn_variables)
    return partition


def compute_modularity(graph, mod_matrix, partition):
    n = mod_matrix.shape[0]
    x = np.array(partition).reshape(n, 1)
    mod = x.transpose() * mod_matrix * x
    return mod.item(0)


# modularity with DWave
def sapi_refine_modularity(
        graph,
        solver,
        hardware_size,  # max size subproblem
        ptn_variables,  # ptn_variables[node] = 0,1,'free'
        num_reads,
        annealing_time,
        embeddings=False,  # if false, get fast embedding
):
    sub_B_matrix, bias, constant = get_sub_mod_matrix(graph, ptn_variables)
    n = sub_B_matrix.shape[0]
    if n > hardware_size:
        print n, hardware_size
        raise ValueError('Number free variables exceeds hardware size')
    coupler = {}
    # we add negative because we maximize modularity
    bias = [-i for i in bias]
    for i in range(n - 1):
        for j in range(i + 1, n):
            coupler[(i, j)] = -sub_B_matrix.item((i, j))
            coupler[(j, i)] = -sub_B_matrix.item((j, i))
    A = get_hardware_adjacency(solver)
    #print "embedding..."
    if not embeddings:
        print 'finding embedding ....'
        embeddings = find_embedding(coupler, A, verbose=0, fast_embedding=True)
    (h0, j0, jc, new_emb) = embed_problem(
        bias, coupler, embeddings, A, clean=True, smear=True)
    emb_j = j0.copy()
    emb_j.update(jc)
    #print "On DWave..."
    result = solve_ising(
        solver, h0, emb_j, num_reads=num_reads, annealing_time=annealing_time)
    #print result
    #print "On DWave...COMPLETE"
    energies = result['energies']
    #print energies
    #print result['solutions']
    #print min(energies), max(energies)
    new_answer = unembed_answer(result['solutions'], new_emb, 'minimize_energy',
                                bias, coupler)
    min_energy = 10**10
    best_soln = []
    for i, ans in enumerate(new_answer):
        soln = ans[0:n]
        assert 3 not in soln
        en = energies[i]
        if en < min_energy:
            #print 'energy', en
            min_energy = en
            best_soln = copy.deepcopy(soln)
    return get_new_partition(best_soln, ptn_variables)


def get_random_nodes(graph, hardware_size):
    return random.sample(graph.nodes(), hardware_size)


def _node_gain(node, mod_matrix, ptn):
    ''' return change in modularity of moving node to new part'''
    int_deg = 0
    ext_deg = 0
    for i in range(mod_matrix.shape[0]):
        if i != node:
            if ptn[i] == ptn[node]:
                int_deg += mod_matrix.item((i, node))
            else:
                ext_deg += mod_matrix.item((i, node))
    #return  -2*int_deg - mod_matrix.item((node, node))
    return 4 * (ext_deg - int_deg)  # min-cut


def verify_gain(node, graph, mod_matrix, ptn):
    mod1 = compute_modularity(graph, mod_matrix, ptn)
    ptn[node] = -ptn[node]
    mod2 = compute_modularity(graph, mod_matrix, ptn)
    ptn[node] = -ptn[node]
    return mod2 - mod1


def get_top_gain_nodes(graph, mod_matrix, ptn, hardware_size):
    ''' return top gain nodes '''
    node_gain = []
    for node in sorted(graph.nodes()):
        #noise = random.random() * 10**(-5)
        noise = 0
        gain = _node_gain(node, mod_matrix, ptn) + noise
        node_gain.append((gain, node))
        #gain2 = verify_gain(node, graph, mod_matrix, ptn)
    sort_gain = sorted(node_gain, reverse=True)

    #gains = [0.25/nx.number_of_edges(graph)*gain for gain, _ in sort_gain]
    #gains = [gain for gain, _ in sort_gain]
    nodes = [node for _, node in sort_gain]
    #print 'top node', nodes[:hardware_size]
    return nodes[:hardware_size]


def get_top_spectral_gain(graph, mod_matrix, ptn, hardware_size, sp_order):
    node_gain = []
    for node in sorted(graph.nodes()):
        gain = _node_gain(node, mod_matrix, ptn)
        node_gain.append((gain, node))
    sort_gain = sorted(node_gain, reverse=True)
    #gains = [0.25/nx.number_of_edges(graph)*gain for gain, _ in sort_gain]
    #print gains
    #nodes = [node for _, node in sort_gain]
    top_node = sort_gain[0][1]
    subset = get_neigborhood(top_node, sp_order, hardware_size)
    return sorted(subset)


def get_free_nodes(graph,
                   mod_matrix,
                   ptn,
                   hardware_size,
                   sp_order,
                   method=False):
    if method == 'top_gain':
        free_nodes = get_top_gain_nodes(graph, mod_matrix, ptn, hardware_size)
    elif method == 'spectral':
        free_nodes = get_top_spectral_gain(graph, mod_matrix, ptn,
                                           hardware_size, sp_order)
    elif method == 'boundary_spectral':
        free_nodes = get_spectral_boundary(graph, mod_matrix, ptn,
                                           hardware_size, sp_order)
    elif method == 'tg_sp_same_part':
        free_nodes = tg_sp_same_part(graph, mod_matrix, ptn, hardware_size,
                                     sp_order)
    else:
        free_nodes = get_random_nodes(graph, hardware_size)
    return sorted(free_nodes)


def data_to_graph(filename):
    return nx.convert_node_labels_to_integers(
        nx.read_edgelist(
            filename, comments='%', data=False, create_using=nx.OrderedGraph()))


def ising_to_file(B_matrix, bias):
    data_var = {}
    data_var['couplers'] = 'set couplers :=\n'
    data_var['nodes'] = 'set nodes :=\n'
    data_var['bias'] = 'param bias := \n'
    data_var['weight'] = 'param w := \n'
    mygraphfile = open('ising.dat', 'w')
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


def pyomo_refine(graph, ptn_variables):
    sub_B_matrix, bias, constant = get_sub_mod_matrix(graph, ptn_variables)
    ising_to_file(sub_B_matrix, bias)
    instance = minimize_ising_model.model.create_instance("ising.dat")
    solver = SolverFactory("gurobi")
    solver.options['mipgap'] = 0.00000001
    solver.options['parallel'] = -1
    solver.options['timelimit'] = 20
    results = solver.solve(instance, tee=False)
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
    new_ptn = get_new_partition(ising_partition, ptn_variables)
    #print new_ptn
    return new_ptn


def get_boundary(graph, ptn):
    boundary = {}
    for u, v in graph.edges():
        if ptn[u] != ptn[v]:
            boundary[u] = ptn[u]
            boundary[v] = ptn[v]
    return boundary


def get_spectral_boundary(graph, mod_matrix, ptn, hardware_size, sp_order):
    '''
    Pick a node
    Get spectral neighborhood of node
    if boundary nodes more than hardware_size, subset returned must all be 
    boundary nodes, else pick other nodes
    
    How to pick a node:
        1. top gain boundary node
        2. random boundary node
    '''
    boundary = get_boundary(graph, ptn)
    node_gain = []
    for node in boundary:
        gain = _node_gain(node, mod_matrix, ptn)
        node_gain.append((gain, node))
    top_node = max(node_gain)[1]


def tg_sp_same_part(graph, mod_matrix, ptn, hardware_size, sp_order):
    node_gain = []
    for node in graph.nodes():
        gain = _node_gain(node, mod_matrix, ptn)
        node_gain.append((gain, node))
    top_node = max(node_gain)[1]
    return spectral_neigh_same_part(top_node, graph, mod_matrix, ptn,
                                    hardware_size, sp_order)


def spectral_neigh_same_part(node, graph, mod_matrix, ptn, hardware_size,
                             sp_order):
    #top_node = random.choice(boundary.keys())
    #boundary_sp_order = [i  for i in sp_order if i in boundary]
    neigh_sp_order = []
    for i in sp_order:
        #if i in boundary:
        if ptn[i] == ptn[node]:
            neigh_sp_order.append(i)
    return get_neigborhood(node, neigh_sp_order, hardware_size)


def get_neigborhood(node, mylist, hardware_size):
    '''
    Get neighboorhood around mylist
    '''
    node_indx = mylist.index(node)
    #n = nx.number_of_nodes(graph)
    n = len(mylist)
    l_indx = node_indx - 1
    r_indx = node_indx + 1
    max_left = max(node_indx - hardware_size / 2, 0)
    max_right = min(node_indx + hardware_size / 2, n)
    if hardware_size / 2 > node_indx:
        # add to right
        add = hardware_size / 2 - node_indx
        max_right = min(max_right + add, n)
    if hardware_size / 2 + node_indx > n:
        add = hardware_size / 2 + node_indx - n
        max_left = max(max_left - add, 0)
    #print 'left right', max_left, node_indx, max_right
    #print 'boundary',  len(mylist[max_left:max_right])
    return sorted(mylist[max_left:max_right])


def list_to_sorted(graph, init_ptn):
    ptn = [0 for i in graph.nodes()]
    for i, node in enumerate(graph.nodes()):
        ptn[node] = init_ptn[i]
    assert 0 not in ptn
    return ptn


def main():
    #method = 'pyomo'
    method = 'dwave'
    free_node_method = 'top_gain'
    #free_node_method = 'spectral'
    #free_node_method = 'random'
    #free_node_method = 'boundary_spectral'
    #free_node_method = 'tg_sp_same_part'
    if method == 'dwave':
        solver = start_sapi()
        embedding = get_native_embedding(solver)
        num_reads = 1000
        annealing_time = 200
    filename = sys.argv[1]
    graph = data_to_graph(filename)
    #print graph.nodes()
    print('%i nodes, %i edges' % (nx.number_of_nodes(graph),
                                  nx.number_of_edges(graph)))
    if nx.number_of_nodes(graph) > 600:
        exit()
    mod_matrix = nx.modularity_matrix(graph, nodelist=sorted(graph.nodes()))
    nnodes = nx.number_of_nodes(graph)
    hardware_size = 25
    #init_ptn = [1 - 2*random.randint(0,1) for _ in range(nnodes)]
    seeds = [
        1070, 8173, 3509, 8887, 1314, 4506, 5219, 3765, 1420, 7778, 3734, 6509,
        1266, 5063, 6496, 4622, 7018, 6052, 8932, 8215, 1254, 400, 3260, 5999,
        1331, 8073, 7357, 2928, 7208, 3874
    ]
    niters = []
    mod_values = []
    for __, seed in enumerate(seeds):
        #print('exp %i' %__)
        #seed = 0
        random.seed(seed)
        np.random.seed(seed)
        init_ptn = [
            1 - 2 * x for x in list(
                np.random.randint(2, size=(graph.number_of_nodes(),)))
        ]
        #print init_ptn
        mod = compute_modularity(graph, mod_matrix, init_ptn)
        #print('init modularity:', mod, 0.25*mod/nx.number_of_edges(graph))
        ptn_variables = {}
        for node in sorted(graph.nodes()):
            ptn_variables[node] = init_ptn[node]
        free_nodes = get_random_nodes(graph, hardware_size)
        free_set = set(free_nodes)
        sp_order = nx.spectral_ordering(graph)
        #sp_order = list(reverse_cuthill_mckee_ordering(graph))
        #print(sp_order)
        free_nodes = get_free_nodes(
            graph,
            mod_matrix,
            init_ptn,
            hardware_size,
            sp_order,
            method=free_node_method)
        not_converge = True
        myiter = 0
        nconv = 5
        best_soln = -float('inf')
        while not_converge:
            myiter += 1
            #print len(free_nodes)
            for node in free_nodes:
                ptn_variables[node] = 'free'
            if method == 'dwave':
                new_ptn = sapi_refine_modularity(graph, solver, hardware_size,
                                                 ptn_variables, num_reads,
                                                 annealing_time, embedding)
            else:
                new_ptn = pyomo_refine(graph, ptn_variables)
            for node in free_nodes:
                ptn_variables[node] = new_ptn[node]

            mod = compute_modularity(graph, mod_matrix, new_ptn)
            #print(myiter, 'refine modularity:', mod, 0.25*mod/nx.number_of_edges(graph))

            free_nodes = get_free_nodes(
                graph,
                mod_matrix,
                new_ptn,
                hardware_size,
                sp_order,
                method=free_node_method)
            current_free_set = set(free_nodes)
            if mod > best_soln:
                best_soln = mod
                best_it = myiter
            if free_set == current_free_set:
                not_converge = False
            elif myiter - best_it >= nconv:
                not_converge = False
            free_set = current_free_set
        niters.append(myiter)
        mod_values.append(0.25 * mod / nx.number_of_edges(graph))
        #print(seed, myiter,  0.25*mod/nx.number_of_edges(graph))
    best = max(mod_values)
    worst = min(mod_values)
    av = np.mean(mod_values)
    std = np.std(mod_values)
    b_it = min(niters)
    w_it = max(niters)
    av_it = np.mean(niters)
    std_it = np.std(niters)
    #print(seeds)
    out = [worst, av, best, std, b_it, av_it, w_it, std_it]
    out = '& '.join([str(round(i, 4)) for i in out])
    print(out)
    print('-------------------\n')
    '''
    for mynode in sorted(graph.nodes()):
        free_nodes = spectral_neigh_same_part(mynode, graph,
                                   mod_matrix, new_ptn, hardware_size, sp_order)
        for node in free_nodes:
            ptn_variables[node] = 'free'
        if method == 'dwave':
            new_ptn = sapi_refine_modularity(graph,
                                                solver,
                                                hardware_size, 
                                                ptn_variables,  
                                                num_reads,
                                                annealing_time,
                                                embedding)
        else:
            new_ptn = pyomo_refine(graph, ptn_variables)
        for node in free_nodes:
            ptn_variables[node] = new_ptn[node]
            
        mod = compute_modularity(graph, mod_matrix, new_ptn)
        print(mynode, 'refine modularity:', mod, 0.25*mod/nx.number_of_edges(graph))
    '''


def main2():
    """ spectral same part"""
    method = 'pyomo'
    if method == 'dwave':
        solver = start_sapi()
        embedding = get_native_embedding(solver)
        num_reads = 1000
        annealing_time = 200
    #seed = random.randint(1,10000)
    #seed = 8834
    #print seed
    random.seed(seed)
    filename = sys.argv[1]
    graph = data_to_graph(filename)
    #print '%i nodes, %i edges' %(nx.number_of_nodes(graph), nx.number_of_edges(graph))
    mod_matrix = nx.modularity_matrix(graph, nodelist=sorted(graph.nodes()))
    nnodes = nx.number_of_nodes(graph)
    hardware_size = 25
    ptn = [1 - 2 * random.randint(0, 1) for _ in range(nnodes)]
    #init_ptn = [1 for _ in range(nnodes)]
    #init_ptn = [np.sign(i) for i in nx.fiedler_vector(graph).tolist()]
    mod = compute_modularity(graph, mod_matrix, ptn)
    #print 'init modularity:', mod, 0.25*mod/nx.number_of_edges(graph)

    ptn_variables = {}
    for node in sorted(graph.nodes()):
        ptn_variables[node] = ptn[node]
    sp_order = nx.spectral_ordering(graph)
    for mynode in sorted(graph.nodes()):
        free_nodes = spectral_neigh_same_part(mynode, graph, mod_matrix, ptn,
                                              hardware_size, sp_order)
        for node in free_nodes:
            ptn_variables[node] = 'free'
        if method == 'dwave':
            new_ptn = sapi_refine_modularity(graph, solver, hardware_size,
                                             ptn_variables, num_reads,
                                             annealing_time, embedding)
        else:
            new_ptn = pyomo_refine(graph, ptn_variables)
        for node in free_nodes:
            ptn_variables[node] = new_ptn[node]

        mod = compute_modularity(graph, mod_matrix, new_ptn)
        #print mynode, 'refine modularity:', mod, 0.25*mod/nx.number_of_edges(graph)


if __name__ == '__main__':
    main()
    #main2()
