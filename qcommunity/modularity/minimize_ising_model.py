"""
Ising Model for pyomo
"""

from pyomo.environ import *
import networkx as nx
import sys

model = AbstractModel()
# Nodes or variables
model.nodes = Set()
# Non zero couplers
model.couplers = Set(within=model.nodes * model.nodes)
# Coupler weights
model.w = Param(model.couplers)
# Bias weights
model.bias = Param(model.nodes)
# Ising variable (transformed to Ising)
model.x = Var(model.nodes, within=Binary)


# Minimize Ising
def ising(model):
    energy = 0
    for u, v in model.couplers:
        if u != v:
            if u > v:
                raise ValueError('Upper triangular elements only')
            energy += 2 * model.w[u, v] * (2 * model.x[u] - 1) * (
                2 * model.x[v] - 1)
        else:
            energy += model.w[u, v]
    for i in model.nodes:
        energy += model.bias[i] * (2 * model.x[i] - 1)

    return energy


model.min_ising = Objective(rule=ising, sense=minimize)


# redundant expression to get read of empty constraints warning
# there is probably a better way
def redundadant(model, node):
    return model.x[node] <= 1


model.boolean = Constraint(model.nodes, rule=redundadant)
