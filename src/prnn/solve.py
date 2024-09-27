"""
Given equations, solves for a reservoir which approximates the system.
* Calls matlab engine to run prnn; see src/prnn/matlab_scripts/runMethod.m for details
"""

# add reservoir.py to import path
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from prnn.reservoir import *
import numpy as np
from typing import List
import sympy as sp
import matlab.engine
from utils import utils


# assumes sym_eqs are passed as an array of sympy equations
def solve_self(self: Reservoir, sym_eqs, verbose: bool = False) -> Reservoir:
    print("Solving for reservoir. This may take a moment!")

    # start and configure matlab engine
    eng = matlab.engine.start_matlab()
    if verbose:
        print("* matlab engine started")

    eng.addpath(r"/Users/witt/all/cncl/compiler/src/matlab_dependencies", nargout=0)
    eng.cd(r"/Users/witt/all/cncl/compiler/src/matlab_dependencies", nargout=0)
    if verbose:
        print("* added scripts to matlab path")

    # convert to matlab format, run
    A, B, r_init, x_init, global_timescale, gamma = self.py2mat()
    matlab_eqs = [sp.octave_code(eq) for eq in sym_eqs]
    A, B, r_init, x_init, d, O, R = eng.runMethod(
        A, B, r_init, x_init, global_timescale, gamma, matlab_eqs, verbose, nargout=7
    )  # add nargout=0 if ignoring output
    eng.quit()

    # solve for W
    O = np.array(O, dtype=float)
    R = np.array(R, dtype=float)
    W, _, _, _ = np.linalg.lstsq(R.T, O.T)
    W = W.T

    return Reservoir.mat2py(
        A, B, r_init, x_init, self.global_timescale, self.gamma, d, W
    )


Reservoir.solve_self = solve_self
