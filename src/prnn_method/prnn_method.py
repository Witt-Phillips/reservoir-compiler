# add reservoir.py to import path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
import numpy as np
from typing import List
import sympy as sp
import matlab.engine

# assumes sym_eqs are passed as an array of sympy equations
def runMethod(self: Reservoir, sym_eqs, inputs: np.ndarray, verbose: bool = False) -> np.ndarray:
    if verbose:
        print("running reservoir...")

    # verify dimensions are valid
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
    eng.addpath(r'/Users/witt/all/cncl/compiler/src/matlab_dependencies', nargout=0)
    eng.addpath(r'/Users/witt/all/cncl/compiler/src/prnn_method/matlab_scripts', nargout=0)
    eng.cd(r'/Users/witt/all/cncl/compiler/src/prnn_method/matlab_scripts', nargout=0)
    if verbose:
        print("* added scripts to matlab path")
    
    # save reservoir to matlab readable format
    A = matlab.double(self.A.tolist())
    B = matlab.double(self.B.tolist())
    r_init = matlab.double(self.r_init.tolist())
    x_init = matlab.double(self.x_init.tolist())
    global_timescale = matlab.double([self.global_timescale])
    gamma = matlab.double([self.gamma])
    matlab_inputs = matlab.double(inputs.tolist())
    
    # convert symbolic equations to strings for matlab evaluation
    matlab_eqs = [sp.octave_code(eq) for eq in sym_eqs]
    # print("octave code: ", matlab_eqs)
    # print(x_init)

    # run matlab script
    A, B, r_init, x_init, global_timescale, gamma, d, W, outputs = eng.runMethod(A, B, r_init, x_init, global_timescale, gamma, matlab_inputs, matlab_eqs, verbose, nargout=9) # add nargout=0 if ignoring output  
    #outputs[np.isnan(outputs)] = 0 # TODO: this necessary?
    eng.quit()

    # convert back to python reservoir
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    W = np.array(W, dtype=float)
    d = np.array(d, dtype=float)
    r_init = np.array(r_init, dtype=float)
    x_init = np.array(x_init, dtype=float)
    global_timescale = float(global_timescale)
    gamma = float(gamma)
    outputs = np.array(outputs, dtype=float)

    reservoir = Reservoir(A, B, r_init, x_init, global_timescale, gamma)
    reservoir.d = d
    return reservoir, W, outputs

Reservoir.runMethod = runMethod