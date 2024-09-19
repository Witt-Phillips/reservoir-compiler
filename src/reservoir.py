import numpy as np
import matlab.engine
import pickle as pkl
import os

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
    def __init__(self, A, B, r_init, x_init, global_timescale=0.1, gamma=100, d=None, W=None):
        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.r_init: np.ndarray = r_init if r_init is not None else np.zeros((A.shape[0], 1))
        self.r: np.ndarray = np.zeros((A.shape[0], 1))
        self.x_init: np.ndarray = x_init
        self.global_timescale: float = global_timescale
        self.gamma: float = gamma

        self.d = d if d is not None else np.arctanh(self.r_init) - (self.A @ self.r_init) - (self.B @ self.x_init) if r_init is not None else np.zeros((A.shape[0], 1))
        self.W = W

        # for circuitry
        self.usedInputs = set()
        self.usedOutputs = set()

    def copy(self):
        return Reservoir(self.A, self.B, self.r_init, self.x_init, self.global_timescale, self.gamma, self.d, self.W)
    
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
    
    def run4input(self, inputs, W=None, verbose=False):
        W = W if W is not None else self.W
        if W is None:
            return ValueError("run4input: W must be defined, either by argument or in reservoir object")

        # Ensure inputs are 4 dimensinoal on z axis
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
        inputs = np.repeat(inputs, 4, axis=2)

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
        if self.W is not None:
            print("W:\n", np.round(self.W, precision))
        print("--------------------")
    
    def printDims(self):
        print("A: ", self.A.shape)
        print("B: ", self.B.shape)
        print("r_init: ", self.r_init.shape)
        print("x_init: ", self.x_init.shape)
        print("d: ", self.d.shape)
        print("r: ", self.r.shape)
        print("global_timescale: ", self.global_timescale)
        print("gamma: ", self.gamma)
        if self.W is not None:
            print("W: ", self.W.shape)

    def save(self, filename, directory="./src/presets"):
        
        # check dir exists
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"The directory '{directory}' does not exist. Please create the directory or navigate to the root 'compiler' directory."
            )
        
        os.makedirs(directory, exist_ok=True)
        
        # save object to res file
        filepath = os.path.join(directory, f"{filename}.rsvr")
        with open(filepath, "wb") as f:
            pkl.dump(self, f)
        
        if os.path.isfile(filepath):
            print("save: created", filepath)
        else:
            raise FileNotFoundError("error: save: dumped file, then couldn't find it")

    
    @classmethod
    def load(cls, filename, directory= "./src/presets"):
        filepath = os.path.join(directory, f"{filename}.rsvr")
        
        # check dir exists
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"The directory '{directory}' does not exist. Please ensure the correct directory structure is in place."
            )
        
        # find file
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"The file '{filepath}' does not exist. Please check the filename or ensure the file has been created in the 'src/presets' directory."
            )
        
        # load reservoir
        with open(filepath, "rb") as f:
            obj = pkl.load(f)
        return obj

    # convert python reservoir to matlab components
    def py2mat(self):
        # save reservoir to matlab readable format
        A = matlab.double(self.A.tolist())
        B = matlab.double(self.B.tolist())
        r_init = matlab.double(self.r_init.tolist())
        x_init = matlab.double(self.x_init.tolist())
        global_timescale = matlab.double([self.global_timescale])
        gamma = matlab.double([self.gamma])
        return A, B, r_init, x_init, global_timescale, gamma

    # Convert matlab components to python reservoir 
    @staticmethod
    def mat2py(A, B, r_init, x_init, global_timescale, gamma, d=None, W=None):
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        d = np.array(d, dtype=float) if d is not None else None
        r_init = np.array(r_init, dtype=float)
        x_init = np.array(x_init, dtype=float)
        global_timescale = float(global_timescale)
        gamma = float(gamma)
        W = np.array(W, dtype=float) if W is not None else None
        return Reservoir(A, B, r_init, x_init, global_timescale, gamma, d, W)
    
    @staticmethod
    def gen_baseRNN(latent_dim, input_dim, global_timescale=0.001, gamma=100):
        np.random.seed(0)
        A = np.zeros((latent_dim, latent_dim))                     # Adjacency matrix 
        B = (np.random.rand(latent_dim, input_dim) - 0.5) * 0.05  # Input weight matrix
        r_init = np.random.rand(latent_dim, 1) - 0.5
        x_init = np.zeros((input_dim, 1))
        return Reservoir(A, B, r_init, x_init, global_timescale, gamma)

    # implemented in prnn/solve.py
    def solve_self(self, sym_eqs, inputs=None, verbose: bool = False) -> np.ndarray:
        pass
    
    # static variant generates baseRNN
    @staticmethod
    def solve(sym_eqs, verbose=False):
        # determine number of baseRNN inputs -- init latents at 10x inputs
        x = set()
        for eq in sym_eqs:
            for symbol in eq.rhs.free_symbols:
                x.add(symbol)
        num_x = len(x)

        #TODO how many latents -- parameter sweep? 
        baseRNN = Reservoir.gen_baseRNN(num_x * 10, num_x)
        #baseRNN = Reservoir.gen_baseRNN(1000, num_x)

        return baseRNN.solve_self(sym_eqs, verbose)


    def doubleOutput(self, output_idx):
        if output_idx <= 0 and output_idx > self.W.shape[0]:
            return ValueError(f"invalid output; cannot double")
        self.W = np.vstack((self.W, self.W[output_idx, :]))
        return self

            
        