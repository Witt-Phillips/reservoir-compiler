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
    def __init__(self, A=None, B=None, r_init=None, x_init=None,
                global_timescale: float=1, local_timescale: float=1):
        self.__dict__.update(A=A, B=B, r_init=r_init, x_init=x_init, global_timescale=global_timescale, local_timescale=local_timescale)
        self.d = np.arctanh(self.r_init) - self.A @ self.r_init - np.dot(self.B, self.x_init)  # calculate bias s.t. r is fixed at x_init
        self.r = np.zeros(self.A.shape[0])  # r: n x 1

    # approximate the diff eq describing r evolution in continuous time via RK4 integration
    # Input: x at current timestep
    # Output: none, updates self.r
    def propagate(self, x):
        k1 = self.global_timescale * self.del_r(self.r, x)
        k2 = self.global_timescale * self.del_r(self.r + k1 / 2, x)
        k3 = self.global_timescale * self.del_r(self.r + k2 / 2, x)
        k4 = self.global_timescale * self.del_r(self.r + k3, x)
        self.r = self.r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return self.r

    def del_r(self, r, x): 
        return self.local_timescale * (-r + np.tanh(np.dot(self.A, r) + np.dot(self.B, x) + self.d))
    
    # implemented in decompile.py
    def decompile(self, order: int):
        pass

    def decompilev2(self, order: int):
        pass

    def print(self):
        print("--------------------")
        print("A:\n", self.A)
        print("B:\n", self.B)
        print("r_init: ", self.r_init)
        print("x_init: ", self.x_init)
        print("d: ", self.d)
        print("r: ", self.r)
        print("global_timescale: ", self.global_timescale)
        print("local_timescale: ", self.local_timescale)
        print("--------------------")

