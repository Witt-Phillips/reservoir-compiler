import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
from utils.plotters import *

def main():
    dt = 0.001
    gam = 100
    N1 = 1000

    A1 = (np.random.rand(N1, N1) - 0.5) * (np.random.rand(N1, N1) < 0.05)
    A1 = csr_matrix(A1 / abs(eigs(A1, k=1, which='LM', maxiter=1e6)[0][0]) * 0.01)
    B1 = (np.random.rand(N1, 3) - 0.5) * 0.2
    rs1 = np.random.rand(N1) - 0.5
    xs1 = np.array([0, 0, 0])
    R1 = Reservoir(A1, B1, rs1, xs1, dt, gam)
    R1.r = rs1
    d1 = R1.d

    # Decompile
    _, C1, C1a = R1.decompose(rs1, A1 @ rs1 + B1 @ xs1 + d1, gam, 3)
    RsNPL1 = np.hstack((C1, C1a.reshape(N1, -1)))
    plt_decomposition(RsNPL1)








if __name__ == "__main__":
    main()