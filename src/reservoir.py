import numpy as np

"""
Reservoir Structure:
    * n = latent dim
    * k = input dim
    * m = output dim
* r: n x 1
* x: k x 1
* d: n x 1
* A: n x n
* B: n x k
* W: m x n
"""
class Reservoir:
    def __init__(self, A, B, r_init, x_init, global_timescale=0.1, gamma=100):
        self.A = A
        self.B = B
        self.r_init = r_init if r_init is not None else np.zeros((A.shape[0], 1))
        self.r = np.zeros((A.shape[0], 1))
        self.x_init = x_init
        self.global_timescale = global_timescale
        self.gamma = gamma
        self.d = np.arctanh(self.r_init) - (self.A @ self.r_init) - (self.B @ self.x_init) if r_init is not None else np.zeros((A.shape[0], 1))
        
        # only defined after running through matlab.
        self.matlabVersion = []
        self.W = []

    def copy(self):
        return Reservoir(self.A, self.B, self.r_init, self.x_init, self.global_timescale, self.gamma)
    
    def del_r(self, r, x):
        # Ensure r and x are column vectors
        r = r.reshape(-1, 1)
        x = x.reshape(-1, 1)
        dr = self.gamma * (-r + np.tanh(self.A @ r + self.B @ x + self.d))
        return dr

    def propagate(self, x):
        # x is expected to be a 3D array with shape (n, 1, 4)
        x = x.reshape(-1, 1, 4)  # Ensure x is in the correct shape

        k1 = self.global_timescale * self.del_r(self.r, x[:, 0, 0].reshape(-1, 1))
        k2 = self.global_timescale * self.del_r(self.r + k1 / 2, x[:, 0, 1].reshape(-1, 1))
        k3 = self.global_timescale * self.del_r(self.r + k2 / 2, x[:, 0, 2].reshape(-1, 1))
        k4 = self.global_timescale * self.del_r(self.r + k3, x[:, 0, 3].reshape(-1, 1))

        self.r = self.r + (k1 + (2 * k2) + 2 * (k3 + k4)) / 6
        return self.r
    
    def run4input(self, W, inputs, verbose=False):
        if verbose:
            print("Running for input...")

        # Ensure inputs are 4 dimensinoal on z axis
        if inputs.shape[2] == 1:
            inputs = np.repeat(inputs, 4, axis=2)
        elif inputs.shape[2] != 4:
            raise ValueError("inputs must be 4 dimensional on z axis")

        nInd = 0
        nx = inputs.shape[1]
        states = np.zeros((self.A.shape[0], nx))
        states[:, 0] = self.r.flatten()
        
        if verbose:
            print("." * 100)
        for i in range(1, nx):
            if i > nInd * nx:
                nInd += 0.01
                if verbose:
                    print("+", end="")
            self.propagate(inputs[:, i - 1, :])
            states[:, i] = self.r.flatten()
        print()

        #print(states[:, 0:4])
        #print(self)
        return W @ states

    
    # methods implemented elsewhere
    def decompile(self, order: int):
        print("decompile: deprecated, use runMethod instead")
        pass
    def compile(self, output_eqs, input_syms, C1, Pd1, PdS, R, verbose=False):
        print("compile: deprecated, use runMethod instead")
        pass
    
    def print(self, precision=2):
        print("--------------------")
        print("Reservoir Parameters")
        print("A:\n", np.round(self.A, precision))
        print("B:\n", np.round(self.B, precision))
        print("r_init:\n", np.round(self.r_init, precision))
        print("x_init:\n", np.round(self.x_init, precision))
        print("d:\n", np.round(self.d, precision))
        print("r:\n", np.round(self.r, precision))
        print("global_timescale: ", np.round(self.global_timescale, precision))
        print("1/gamma: ", np.round(1/self.gamma, precision))
        print("--------------------")

def gen_baseRNN(latent_dim, input_dim, global_timescale=0.001, gamma=100):
    np.random.seed(0)
    A = np.zeros((latent_dim, latent_dim))                     # Adjacency matrix 
    B = (np.random.rand(latent_dim, input_dim) - 0.5) * 0.05  # Input weight matrix
    r_init = np.random.rand(latent_dim, 1) - 0.5
    x_init = np.zeros((input_dim, 1))
    return Reservoir(A, B, r_init, x_init, global_timescale, gamma)

# runMethod variant -- generates baseRNN.
def runMethod(sym_eqs, inputs, verbose=False):
    # determine number of baseRNN inputs -- init latents at 10x inputs
    x = set()
    for eq in sym_eqs:
        for symbol in eq.rhs.free_symbols:
            x.add(symbol)
    num_x = len(x)

    baseRNN = gen_baseRNN(num_x * 10, num_x)
    return baseRNN.runMethod(sym_eqs, inputs, verbose)