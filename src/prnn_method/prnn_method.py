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
def solveReservoir(self: Reservoir, sym_eqs, inputs=None, verbose: bool = False) -> np.ndarray:
    print("Solving for reservoir. This may take a moment!")

    if verbose:
        print("running reservoir...")

    # verify dimensions are valid
    if inputs is not None:
        sigs = inputs.shape[0]
        res_inps = self.B.shape[1]
        recs = 1 # TODO: iterate through sym_eqs, count recurrences (sym defined as dx# instead of o#)
        outs =  1 #len(sym_eqs) - recs # TODO: after matlab has run, make sure we have the right number of outputs

        preface = "runMethod: reservoir compliance: "
        assert sigs <= res_inps, preface + f"more signals ({sigs}) than expected inputs ({res_inps})."
        assert recs == res_inps - sigs, preface + "recurrencies must be equal to inputs - signals"

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
    
    matlab_inputs = matlab.double(inputs.tolist()) if inputs is not None else False
    matlab_eqs = [sp.octave_code(eq) for eq in sym_eqs]

    # run matlab script
    A, B, r_init, x_init, global_timescale, gamma, d, W, outputs = eng.runMethod(A, B, r_init, x_init, global_timescale, gamma, matlab_inputs, matlab_eqs, verbose, nargout=9) # add nargout=0 if ignoring output  
    eng.quit()

    # convert back to python format
    reservoir = mat2py(A, B, r_init, x_init, global_timescale, gamma, d, W)
    outputs = np.array(outputs, dtype=float)

    return reservoir, outputs

Reservoir.solveReservoir = solveReservoir