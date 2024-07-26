from reservoir import *
from decompile_matlab import decompile_matlab
from compile_matlab import compile_matlab
from run_matlab import run_matlab
import matplotlib.pyplot as plt
from utils.utils import matrix_mat2np
import numpy as np

def main():
    # Generate a base RNN
    baseRNN: Reservoir = gen_baseRNN(30, 3)
    baseRNN.print(precision=4)
    # Decompile the base RNN
    R, C1, Pd1, PdS = baseRNN.decompile(4, verbose=True)
    # Compile the base RNN
    W = baseRNN.compile(1, 1, C1, Pd1, PdS, R, verbose=True)
    # Redefine the reservoir with recurrency
    """ reccurentRes = Reservoir(baseRNN.A + np.outer(baseRNN.B[:, 2], W), baseRNN.B[:, :2], baseRNN.r_init, np.zeros((2, 1)), baseRNN.global_timescale, baseRNN.gamma)
    reccurentRes.print(precision=4) """

    #define some input set
    time = 1000
    ot = np.ones((2, time))
    pattern1 = np.array([[-.1], [-.1]]) * ot
    pattern2 = np.array([[-.1], [.1]]) * ot
    pattern3 = np.array([[.1], [-.1]]) * ot
    pattern4 = np.array([[.1], [.1]]) * ot
    pt = np.concatenate((pattern1, pattern2, pattern3, pattern4), axis=1)

    # run reservoir for input set
    wrp = baseRNN.run4input(pt, W)
    wrp = matrix_mat2np(wrp)
    print(wrp.shape)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Define time array
    time_array = np.arange(pt.shape[1])

    # Plot input signals
    for i in range(2):
        axes[i].plot(time_array, pt[i, :], label=f'Input {i + 1}')
        axes[i].set_title(f'Input {i + 1}')
        axes[i].legend()
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')

    # Plot output signal
    axes[2].plot(time_array, wrp[0, :], label='Output')
    axes[2].set_title('Output')
    axes[2].legend()
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()