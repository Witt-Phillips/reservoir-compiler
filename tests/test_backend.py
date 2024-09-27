""" 
Runs full backend stack (sym eqs -> reservoir) and checks for well formedness.
"""

import pytest
import numpy as np
import sympy as sp
from examples.imports import Reservoir, inputs

def compare_reservoirs(res1: Reservoir, res2: Reservoir) -> bool:
    """
    Compare two Reservoir objects by specific fields.
    Returns True if all fields are equal, False otherwise.
    """
    # Compare numpy arrays using np.array_equal for the matrix attributes
    if not np.array_equal(res1.A, res2.A):
        return False
    if not np.array_equal(res1.B, res2.B):
        return False
    if not np.array_equal(res1.r_init, res2.r_init):
        return False
    if not np.array_equal(res1.r, res2.r):
        return False
    if not np.array_equal(res1.x_init, res2.x_init):
        return False

    # Compare the scalar attributes
    if res1.global_timescale != res2.global_timescale:
        return False
    if res1.gamma != res2.gamma:
        return False

    return True


def test_nand_gate():
    # Set up symbols
    o1, s1, s2 = sp.symbols("o1 s1 s2")

    # parameters and pitchfork base
    xw = 0.025
    xf = 0.1
    cx = 3 / 13
    ax = -cx / (3 * xw**2)
    pitchfork_bifurcation = (ax * (o1**3)) + (cx * o1)

    # shifting logic for each gate
    logic = {
        "and": -0.1 + (s1 + xf) * (s2 + xf) / (2 * xf),
    }

    # Define equations for the 'and' gate
    logic_eqs = [
        sp.Eq(o1, 0.1 * (pitchfork_bifurcation + logic["and"])),
    ]

    reservoir = Reservoir.solve(logic_eqs)
    reference_res = Reservoir.load("and")

    assert compare_reservoirs(
        reservoir, reference_res
    ), "Failed to properly generate 'AND' gate"
