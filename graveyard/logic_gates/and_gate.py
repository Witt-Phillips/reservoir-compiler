# fix imports
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import sympy as sp
from reservoir import Reservoir

# set rng seed
np.random.seed(0)

def main():
    # Logic gate params
    m = 3 # number of inputs. 2 signal, 1 recurrent
    n = 30 # number of reservoir nodes

    # Reservoir params
    dt = 0.001
    gam = 100
    A = np.zeros((n, n))                     # Adjacency matrix 
    B = (np.random.rand(n, m) - 0.5) * 0.05  # Input weight matrix
    rs = np.random.rand(n, 1) - 0.5
    xs = np.zeros((m, 1))
    reservoir = Reservoir(A, B, rs, xs, dt, gam)
    reservoir.print()

    # Retrieve Decompiled Reservoir
    path = 'src/logic_gates/and_decomp.csv'
    C1 = np.loadtxt(path, delimiter=',')

    


"""     # Write 'program'
    # Define symbolic dynamics
    xw, xf, cx, t = sp.symbols('xw xf cx t', real=True)
    ax = -cx/(3*xw**2)
    x = sp.Function('x')(t)
    dx = [None] * 6
    dxb = cx*x.diff(t, 3) + ax*x.diff(t, 3)**3
    dx[0] = dxb + -0.1 + (x + xf)*(x.diff(t) + xf)/(2*xf)  # AND
    dx[1] = dxb + 0.1 + (x + xf)*(-x.diff(t) - xf)/(2*xf)  # NAND
    dx[2] = dxb + 0.1 + (x - xf)*(-x.diff(t) + xf)/(2*xf)  # OR
    dx[3] = dxb + -0.1 + (x - xf)*(x.diff(t) - xf)/(2*xf)  # NOR
    dx[4] = dxb + 0.0 + (-x)*(x.diff(t))/(xf)  # XOR
    dx[5] = dxb + 0.0 + (x)*(x.diff(t))/(xf)  # XNOR

    print("dx[0]:", dx[0]) """


    
if __name__ == "__main__":
    main()