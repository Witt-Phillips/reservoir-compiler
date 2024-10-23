""" 
Runs full backend stack (sym eqs -> reservoir) and checks for well formedness.
"""

import pytest
import numpy as np
import sympy as sp
from _prnn.reservoir import Reservoir


def compare_reservoirs(res1: Reservoir, res2: Reservoir, tol: float = 1e-8) -> bool:
    """
    Compare two Reservoir objects by specific fields with a small tolerance.
    Returns True if all fields are equal within the tolerance, False otherwise.
    """
    assert np.allclose(
        res1.A, res2.A, atol=tol
    ), "Field 'A' does not match between reservoirs."
    assert np.allclose(
        res1.B, res2.B, atol=tol
    ), "Field 'B' does not match between reservoirs."
    assert np.allclose(
        res1.r_init, res2.r_init, atol=tol
    ), "Field 'r_init' does not match between reservoirs."
    assert np.allclose(
        res1.x_init, res2.x_init, atol=tol
    ), "Field 'x_init' does not match between reservoirs."

    # If all comparisons pass, return True
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

    assert (
        compare_reservoirs(reservoir, reference_res) is True
    ), "Failed to properly generate 'AND' gate"

