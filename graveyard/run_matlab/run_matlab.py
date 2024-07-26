# add reservoir.py to import path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
import numpy as np
import matlab.engine
from utils.utils import matrix_mat2np

def run4input(self: Reservoir, inputs, W, verbose=False):

    print("running reservoir for input...")
    # reservoir dims
    n = self.A.shape[0]
    m = self.B.shape[1]

    # start and configure matlab engine to run decompilation script
    eng = matlab.engine.start_matlab()
    if verbose:
        print("matlab engine started")
    eng.addpath(r'/Users/witt/all/cncl/compiler/pyrnn_compiler/src/run_matlab/matlab_scripts', nargout=0)
    eng.cd(r'/Users/witt/all/cncl/compiler/pyrnn_compiler/src/run_matlab/matlab_scripts', nargout=0)
    if verbose:
        print("added scripts to matlab path")
    
    # save reservoir to matlab readable format
    A = matlab.double(self.A.tolist())
    B = matlab.double(self.B.tolist())
    r_init = matlab.double(self.r_init.tolist())
    x_init = matlab.double(self.x_init.tolist())
    global_timescale = matlab.double([self.global_timescale])
    gamma = matlab.double([self.gamma])
    d = matlab.double(self.d.tolist())
    matlab_inputs = matlab.double(inputs.tolist())

    rp = eng.run_matlab(A, B, r_init, x_init, d, global_timescale, gamma, matlab_inputs, W, nargout=1)
    if verbose:
        print("rp:", rp)
    return rp

Reservoir.run4input = run4input
