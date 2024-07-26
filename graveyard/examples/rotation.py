import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import lstsq


# fix imports
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
from utils.plotters import *

def main():
    # initialize params
    dt = 0.001
    gam = 100
    N1 = 10 # 1000 in Jason's paper
    A1 = (np.random.rand(N1, N1) - 0.5) * (np.random.rand(N1, N1) < 0.05)
    A1 = csr_matrix(A1 / abs(eigs(A1, k=1, which='LM', maxiter=1e6)[0][0]) * 0.01).toarray()
    B1 = (np.random.rand(N1, 3) - 0.5) * 0.2
    rs1 = np.random.rand(N1) - 0.5
    xs1 = np.array([0, 0, 0])
    R1 = Reservoir(A1, B1, rs1, xs1, dt, gam)
    R1.r = rs1
    d1 = R1.d

    # Decompile
    _, C1, C1a, *_ = R1.decompile(rs1, A1 @ rs1 + B1 @ xs1 + d1, gam, 4)
    
    #debug decompiler dims
    print("C1 shape", C1.shape)
    print("C1a shape", C1a.shape)

    RsNPL1 = np.hstack((C1, C1a.reshape(N1, -1)))

    # source code: pi/2 rotation
    Rz = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    OsNPL1 = np.zeros((3, C1.shape[1] * 4))
    OsNPL1[:, 1:4] = Rz
    print(OsNPL1)

    # compile

    #debug dims
    # Print the dimensions of RsNPL1
    print("RsNPL1 shape:", RsNPL1.shape)
    print("OsNPL1 shape", OsNPL1.shape)

    W1_T, residuals, rank, s = lstsq(RsNPL1.T, OsNPL1.T)
    W1 = W1_T.T

    compiler_residual = np.linalg.norm(np.dot(W1, RsNPL1) - OsNPL1, 1)
    print(f"Compiler residual: {compiler_residual}")


if __name__ == "__main__":
    main()