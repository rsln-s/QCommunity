#!/usr/bin/env python
# Justin's shot at optimizing QAOA parameters


def optimize_obj(obj_val, params=None):
    beta = 0.5
    gamma = 0.7
    return (
        beta, gamma, obj_val(beta, gamma)
    )  # return some optimization trace. It will eventually go into optimize.optimize_modularity, so it should at least contain optimal angles
