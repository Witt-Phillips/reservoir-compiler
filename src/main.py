from reservoir import Reservoir
from utils.plotters import *
from decomposition.decompile import *
from misc.manual_decompile import *

def main():
    np.set_printoptions(precision=2)
    # Initialize the matrices and variables with hardcoded values
    n = 5
    k = 3

    A = np.zeros((n, n))
    B = np.random.randn(n, k * 2)
    r_init = np.clip(np.random.randn(n), -0.999999, 0.999999)
    x_init = np.zeros(k * 2)
    global_timescale = 0.1
    local_timescale = 1.0

    reservoir = Reservoir(A, B, r_init, x_init, global_timescale, local_timescale)
    reservoir.decompile(2)
    #manual_decompile(reservoir, 2, verbose=True)


def gen_baseRNN():
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

    return Reservoir(A, B, r_init, x_init, global_timescale, local_timescale)

def decompile2csv(reservoir: Reservoir, order):
    R, _ = reservoir.decompile(order, verbose=True)
    R = np.round(R, 2)
    np.savetxt('src/utils/comparison_outputs/RsNPL1_python.csv', R, delimiter=',', fmt='%.2f')

if __name__ == "__main__":
    main()