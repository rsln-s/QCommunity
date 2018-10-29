import networkx as nx
import random, copy
import sys, os
import numpy as np


def graph2dat(graph):
    folder = ""
    mygraphfile = open(folder + "neg_modularity.dat", 'w')
    data_var = {}
    data_var['couplers'] = 'set couplers :=\n'
    data_var['nodes'] = 'set nodes :=\n'
    data_var['bias'] = 'param bias := \n'
    data_var['weight'] = 'param w := \n'

    modularity_mat = nx.modularity_matrix(graph, nodelist=sorted(graph.nodes()))
    n = nx.number_of_nodes(graph)
    for i in range(n - 1):
        for j in range(i, n):
            w = -modularity_mat.item((i, j))  # neg value
            data_var['couplers'] += ' '.join([str(i), str(j), '\n'])
            data_var['weight'] += ' '.join([str(i), str(j), str(w), '\n'])
    i, j = n - 1, n - 1
    w = -modularity_mat.item((i, j))  # neg value
    data_var['couplers'] += ' '.join([str(i), str(j), '\n'])
    data_var['weight'] += ' '.join([str(i), str(j), str(w), '\n'])
    for i in range(n):
        data_var['nodes'] += str(i) + '\n'
        data_var['bias'] += str(i) + ' 0\n'  # no bias in modularity

    data_var['nodes'] += ';\n'
    data_var['bias'] += ';\n'
    data_var['weight'] += ';\n'
    data_var['couplers'] += ';\n'

    for item in data_var:
        mygraphfile.write(data_var[item])


def data_to_graph(filename):
    graph = nx.Graph()
    edges = np.genfromtxt(filename, comments='%', dtype=int)[:, 0:2]
    for u, v in edges:
        graph.add_edge(u, v)
    mapping = dict(zip(graph.nodes(), range(nx.number_of_nodes(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    return graph


if __name__ == '__main__':
    filename = sys.argv[1]
    #graph = nx.karate_club_graph()
    graph = data_to_graph(filename)
    graph2dat(graph)
