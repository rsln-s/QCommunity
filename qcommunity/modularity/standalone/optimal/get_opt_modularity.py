from pyomo.environ import *
import qcommunity.modularity.standalone.optimal.minimize_ising_model as minimize_ising_model
import sys, os
import networkx as nx
import random
import re
import numpy as np
import multiprocessing
import argparse
import qcommunity.modularity.graphs as gm
from qcommunity.modularity.standalone.optimal.create_data import graph2dat
from qcommunity.utils.import_graph import generate_graph, import_pajek


# Create .dat file
def create_dat(folder):
    graphfile = sys.argv[1]
    epsilon = sys.argv[2]
    folder = "~/dwave/dwave/optimal_gp/"
    pyfile = folder + "create_data.py"
    os.system("python " + pyfile + " " + graphfile + " " + epsilon)


# Partition graph
def get_modularity(graph):
    #graphfile = folder + sys.argv[1]
    #graph = nx.read_graphml(graphfile, node_type=int)
    instance = minimize_ising_model.model.create_instance("neg_modularity.dat")
    solver = SolverFactory("gurobi")
    #solver_manager = SolverManagerFactory('neos') # For Neos server
    # cplex options
    #solver.options['parallel'] = -1
    # gurobi options
    solver.options['threads'] = min(16, multiprocessing.cpu_count())
    #solver.options['timelimit'] = 10

    #results = solver_manager.solve(instance, opt=solver, tee=True) # Neos server
    results = solver.solve(instance, tee=True)
    energy = instance.min_ising()
    solver_modularity = -0.25 * energy / nx.number_of_edges(graph)
    unscaled = -energy
    # Get partition
    varobject = getattr(instance, 'x')
    part0 = []
    part1 = []

    ising_partition = ['unset' for i in graph.nodes()]
    for index in sorted(varobject):
        if varobject[index].value > 0.001:
            part1.append(index)
            ising_partition[index] = 1
        else:
            ising_partition[index] = -1
            part0.append(index)
    #print(ising_partition)
    mod_matrix = nx.modularity_matrix(graph)
    mymod = compute_modularity(graph, mod_matrix, ising_partition)
    res = {
        'unscaled': unscaled,
        'solver_modularity': solver_modularity,
        'from_part': mymod
    }
    print('unscaled:', res['unscaled'])
    print("Solver Modularity:", res['solver_modularity'])
    print("Modularity from part:", res['from_part'])

    return res
    #ising_part2file(folder,ising_partition)


# Partition to file
def ising_part2file(folder, ising_partition):
    myfile = open(folder + "ising_part.txt", "w")
    out = "\n".join([str(i) for i in ising_partition])
    myfile.write(out)
    myfile.close()


def compute_modularity(graph, mod_matrix, partition):
    n = mod_matrix.shape[0]
    x = np.array(partition).reshape(n, 1)
    scale = 0.25 / nx.number_of_edges(graph)
    #scale = 1
    mod = x.transpose() * mod_matrix * x
    return mod.item(0) * scale


def data_to_graph(filename):
    graph = nx.Graph()
    edges = np.genfromtxt(filename, comments='%', dtype=int)[:, 0:2]
    for u, v in edges:
        graph.add_edge(u, v)
    mapping = dict(zip(graph.nodes(), range(nx.number_of_nodes(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def opt_modularity_wrapper(graph_generator_name, left, right, seed=None):
    """
    This function takes the same parameters as get_obj_val and can be imported anywhere and 
    used to get optimal modularity for a graph.
    """
    graph, _ = generate_graph(graph_generator_name, left, right, seed=seed)
    print('%i nodes, %i edges' % (nx.number_of_nodes(graph),
                                  nx.number_of_edges(graph)))
    print('creating dat file...')
    graph2dat(graph)
    print('creating dat file...DONE')
    return get_modularity(graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        type=int,
        default=3,
        help="number of vtx in the left (first) community")
    parser.add_argument(
        "-g",
        "--graph-generator",
        type=str,
        default="get_barbell_graph",
        help="graph generator function")
    parser.add_argument(
        "-r",
        type=int,
        default=3,
        help="number of vtx in the right (second) community")
    parser.add_argument(
        "--graph",
        type=str,
        help="path to KONECT edgelist (out.graphname file)")
    parser.add_argument(
        "--pajek", type=str, help="path to graph in pajek format")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed, only used for graph generator")
    args = parser.parse_args()

    if args.graph:
        # load from edgelist
        graph = data_to_graph(args.graph)
    elif args.pajek:
        graph = import_pajek(args.pajek)
    else:
        # generate
        graph, _ = generate_graph(
            args.graph_generator, args.l, args.r, seed=args.seed)

    print('%i nodes, %i edges' % (nx.number_of_nodes(graph),
                                  nx.number_of_edges(graph)))
    print('creating dat file...')
    graph2dat(graph)
    print('creating dat file...DONE')
    #graph = nx.karate_club_graph()

    get_modularity(graph)
