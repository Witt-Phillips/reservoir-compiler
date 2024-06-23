from reservoir import Reservoir
from utils.plotters import *
from decompile import *

def main():
    np.set_printoptions(precision=2)
    np.random.seed(0)
    # initialize reservoir
    n = 5
    k = 3
    #A = np.random.randn(n, n)
    A = np.zeros((n, n))
    B = np.random.randn(n, k)
    r_init = np.clip(np.random.randn(n), -0.999999, 0.999999)
    x_init = np.zeros(k)
    global_timescale = 0.1
    local_timescale = 1.0

    reservoir = Reservoir(A, B, r_init, x_init, global_timescale, local_timescale)
        
    # Decompile
    # print(tanh_derivs(2))
    # bases = gen_bases([1, 2, 3], 2)
    # print(bases)
    reservoir.decompile(1, verbose=True)

if __name__ == "__main__":
    main()
