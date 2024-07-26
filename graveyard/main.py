from reservoir import Reservoir
from utils.plotters import *
from utils.comp_matrix import *
from misc.manual_decompile import *

def main():
    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    # Initialize the matrices and variables with hardcoded values
    n = 5
    k = 3

    # A = np.zeros((n, n))
    # B = np.random.randn(n, k * 2)
    # r_init = np.clip(np.random.randn(n), -0.999999, 0.999999)
    # x_init = np.zeros(k * 2)
    # global_timescale = 0.1
    # local_timescale = 1.0

   # A Matrix: 5x5 matrix filled with zeros
    A = np.zeros((5, 5))

    # B Matrix: 5x6 matrix with "random" values (manually chosen here for illustration)
    B = np.array([
        [-1.23, 0.66, -0.21, 1.23, -0.88, 0.75],
        [0.58, -1.07, 1.64, -0.31, 0.14, 0.97],
        [0.28, 0.46, 1.52, -1.22, -0.17, -0.88],
        [0.77, -0.76, 0.31, 0.58, -1.10, -0.25],
        [1.33, -0.21, -0.90, 1.44, -0.42, 0.33]
    ])

    # r_init Vector: 5-element vector with "random" values, clipped between -0.999999 and 0.999999
    r_init = np.array([-0.5, 0.5, -0.3, 0.5, -0.2])

    # x_init Vector: 6-element vector filled with zeros
    x_init = np.zeros(6)

    # Global and Local Timescales
    global_timescale = 0.1
    local_timescale = 1.0

    reservoir = Reservoir(A, B, r_init, x_init, global_timescale, local_timescale)
    decompile2csv(reservoir, 2)
    #manual_decompile(reservoir, 2, verbose=True)

    #symbolic_tseries(6, 2)


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
    R, _ = reservoir.decompile(order)
    R = np.round(R, 2)
    np.savetxt('src/utils/comparison_outputs/C1_python.csv', R, delimiter=',', fmt='%.2f')


if __name__ == "__main__":
    main()