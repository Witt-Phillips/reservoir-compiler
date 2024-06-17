from reservoir import Reservoir
from utils.plotters import *

# Example usage
def main():
    n = 5
    k = 3
    A = np.random.randn(n, n)
    B = np.random.randn(n, k)
    r_init = np.clip(np.random.randn(n), -0.999999, 0.999999)
    x_init = np.random.randn(k)
    global_timescale = 0.1
    local_timescale = 1.0
    rs = np.random.randn(n)
    dv = np.random.randn(n)
    gam = 1.0
    o = 4

    reservoir = Reservoir(A, B, r_init, x_init, global_timescale, local_timescale)
    
    # Run the reservoir forward one time step with a random input
    x = np.random.randn(k)
    reservoir.propagate(x)

    # Decomposition
    Pd1, C1, C2, C3a, C3b, C4a, C4b, C4c = reservoir.decompose(rs, dv, gam, o)
    # Concat decomposition matrix (TODO: merge with decompose())
    RsNPL1 = np.hstack((C1, C2.reshape(C1.shape[0], -1)))
    plt_rsnpl1(RsNPL1)

    # plt_decomposition(C1, C2, C3a, C3b, C4a, C4b, C4c)

if __name__ == "__main__":
    main()
