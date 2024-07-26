# add reservoir.py to import path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
import numpy as np
import matlab.engine
from utils.utils import matrix_mat2np

def decompile(self: Reservoir, order: int, verbose=False):
    if verbose:
        print("--------------------\ndecompiling reservoir...")
    else:
        print("decompiling reservoir...")

    # reservoir dims
    n = self.A.shape[0]
    m = self.B.shape[1]

    # start and configure matlab engine to run decompilation script
    eng = matlab.engine.start_matlab()
    if verbose:
        print("matlab engine started")
    eng.addpath(r'/Users/witt/all/cncl/compiler/pyrnn_compiler/src/decompile_matlab/matlab_scripts', nargout=0)
    eng.cd(r'/Users/witt/all/cncl/compiler/pyrnn_compiler/src/decompile_matlab/matlab_scripts', nargout=0)
    if verbose:
        print("added scripts to matlab path")
    
    # save reservoir to matlab readable format
    A = matlab.double(self.A.tolist())
    B = matlab.double(self.B.tolist())
    r_init = matlab.double(self.r_init.tolist())
    x_init = matlab.double(self.x_init.tolist())
    global_timescale = matlab.double([self.global_timescale])
    gamma = matlab.double([self.gamma])
    matlab_order = matlab.double([order])

    #matlab_reservoir = eng.ReservoirTanhB(A, B, r_init, x_init, global_timescale, gamma)
    #print(type(matlab_reservoir))
    # run matlab script to decompile
    R, C1, Pd1, PdS = eng.decompile2py(A, B, r_init, x_init, global_timescale, gamma, matlab_order, nargout=4) # add nargout=0 if ignoring output    
    if verbose: 
        np.set_printoptions(linewidth=400, precision=4)
        pre = 3
        print("dim R:", np.shape(R))
        print("R:\n", np.round(R, pre))
        print("dim C1:", np.shape(C1))
        print("C1:\n", np.round(C1, pre))
        print("dim Pd1:", np.shape(Pd1))
        print("Pd1:\n", np.round(Pd1, pre))
        print("dim PdS:", np.shape(PdS))
        print("PdS:\n", np.round(PdS, pre))
    eng.quit()

    if verbose:
        print("decompiled reservoir via matlab script\n--------------------")
    return R, C1, Pd1, PdS

Reservoir.decompile = decompile