# add reservoir.py to import path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
import numpy as np
import matlab.engine

def compile(self: Reservoir, output_eqs, input_syms, C1, Pd1, PdS, R, verbose=False):
    if verbose:
        print("--------------------\ncompiling reservoir...")
    # reservoir dims
    n = self.A.shape[0]
    m = self.B.shape[1]

    # sanity check
    print("C1 type: ", type(C1))
    print("Pd1 type: ", type(Pd1))
    print("PdS type: ", type(PdS))

    # start and configure matlab engine to run decompilation script
    eng = matlab.engine.start_matlab()
    if verbose:
        print("matlab engine started")
    eng.addpath(r'/Users/witt/all/cncl/compiler/pyrnn_compiler/src/compile_matlab/matlab_scripts', nargout=0)
    eng.cd(r'/Users/witt/all/cncl/compiler/pyrnn_compiler/src/compile_matlab/matlab_scripts', nargout=0)
    if verbose:
        print("added scripts to matlab path")
  
    # save reservoir to matlab readable format
    A = matlab.double(self.A.tolist())
    B = matlab.double(self.B.tolist())
    rs = matlab.double(self.r_init.tolist())
    xs = matlab.double(self.x_init.tolist())
    delT = matlab.double([self.global_timescale])
    gam = matlab.double([self.gamma])

    # save decomposition to matlab readable format
    """ Pd1 = matlab.double(Pd1.tolist())
    PdS = matlab.double(PdS.tolist())
    R = matlab.double(R.tolist()) """

    #matlab_reservoir = eng.ReservoirTanhB(A, B, r_init, x_init, global_timescale, gamma)
    #print(matlab_reservoir)

    # compile via matlab script
    W = eng.compile2py(A, B, rs, xs, delT, gam, 1, 1, C1, Pd1, PdS, R)
    eng.quit()
 
    if verbose:
        print("W: ", np.round(W, 2))
        print("compiled reservoir via matlab script\n--------------------")
    return W

Reservoir.compile = compile