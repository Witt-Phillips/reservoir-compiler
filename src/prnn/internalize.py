# add reservoir.py to import path
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np
from prnn.reservoir import Reservoir


def internalize_recurrences(r: Reservoir, recurrences):
    n = r.A.shape[1]
    m = r.x_init.shape[0]
    newA = np.zeros((n, n))
    newX = np.zeros((m, 1))
    newB = r.B

    for recurrence in recurrences:
        out, inp = recurrence
        newA = newA + r.B[:, inp] @ r.W[out, :]

        if newB.shape[1] > 1:
            newB = np.delete(newB, inp, axis=1)
            newX = np.delete(newX, inp, axis=0)
        else:
            newB = np.zeros((n, 1))
            newX = np.zeros((1, 1))

    r.A = newA
    r.x_init = newX
    r.x = newX
    r.B = newB
    return r
