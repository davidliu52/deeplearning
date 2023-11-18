# -*- coding: utf-8 -*-


import numpy as np


def gradient_descent(f, df, x0: float, lr: float = 0.01):

    reltol = 10**(-4)  # improvement of function value

    # initialization
    converged, step = False, 0
    x, x_steps = x0, [x0]

    while (not converged) and (step < 500):
        y = f(x)                    # current function value
        x = x - lr * df(x)          # upate
        y_next = f(x)               # new function value

        # relative convergence criterion
        if np.abs(y - y_next) < reltol:
            converged = True

        step += 1
        x_steps.append(x)

    x_min = x_steps[-1]
    return x_min, np.array(x_steps)
