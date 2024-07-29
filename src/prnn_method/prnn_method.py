# add reservoir.py to import path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import *
import numpy as np
from typing import List
import sympy as sp
import matlab.engine

# assumes sym_eqs are passed as an array of sympy equations
def solve(self: Reservoir, sym_eqs, verbose: bool = False) -> np.ndarray:
    print("Solving for reservoir. This may take a moment!")

    if verbose:
        print("running reservoir...")

    # reservoir dims
    n = self.A.shape[0]
    m = self.B.shape[1]

    # start and configure matlab engine to run decompilation script
    eng = matlab.engine.start_matlab()
    if verbose:
        print("* matlab engine started")
    if 0:
        eng.addpath(r'/Users/witt/all/cncl/compiler/src/matlab_dependencies', nargout=0)
        eng.addpath(r'/Users/witt/all/cncl/compiler/src/prnn_method/matlab_scripts', nargout=0)
    eng.cd(r'/Users/witt/all/cncl/compiler/src/prnn_method/matlab_scripts', nargout=0)
    if verbose:
        print("* added scripts to matlab path")
    
    # convert reservoir, inputs, and equations to matlab format
    A, B, r_init, x_init, global_timescale, gamma = self.py2mat()
    
    matlab_eqs = [sp.octave_code(eq) for eq in sym_eqs]

    # run matlab script
    A, B, r_init, x_init, global_timescale, gamma, d, W = eng.runMethod(A, B, r_init, x_init, global_timescale, gamma, matlab_eqs, verbose, nargout=8) # add nargout=0 if ignoring output  
    eng.quit()

    return Reservoir.mat2py(A, B, r_init, x_init, global_timescale, gamma, d, W)

Reservoir.solve = solve