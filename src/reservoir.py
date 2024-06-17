import numpy as np
import sympy as sp
from scipy.linalg import inv
import math

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
    def __init__(self, A, B, r_init, x_init, global_timescale, local_timescale):
        self.A = A
        self.B = B
        self.r_init = r_init
        self.x_init = x_init
        self.d = np.arctanh(r_init) - np.dot(A, r_init) - np.dot(B, x_init)  # calculate bias s.t. r is fixed at x_init
        self.r = np.zeros(A.shape[0])  # r: n x 1
        self.global_timescale = global_timescale
        self.local_timescale = local_timescale

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

    # Calculate derivative of r given x.
    def del_r(self, r, x): 
        return self.local_timescale * (-r + np.tanh(np.dot(self.A, r) + np.dot(self.B, x) + self.d))
    
    # Decompose into symbolic bases
    def decompose(self, rs, dv, gam, o):
        N = self.A.shape[0]
        k = self.B.shape[1]

        # Grid indices
        v = np.eye(k).reshape(1, k, k)
        Pd1 = np.eye(k)

        for i in range(2, o):
            Pdp = (Pd1[:, np.newaxis, :] + v).reshape(-1, k)
            Pd1 = np.unique(np.vstack([Pd1, Pdp]), axis=0)

        Pd1 = np.vstack([np.zeros(k), Pd1])
        Pd1 = Pd1[np.lexsort((Pd1.sum(axis=1), Pd1.max(axis=1)))]

        # Initial coefficients
        Ars = np.dot(self.A, rs)

        # Compute higher order B terms
        Bk = []
        Bc = []
        for i in range(1, o):
            PdI = np.where(Pd1.sum(axis=1) == i)[0]
            Bk.append(np.zeros((N, len(PdI))))
            Bc.append(np.zeros(len(PdI)))
            for j in range(len(PdI)):
                Bk[-1][:, j] = np.prod(self.B ** Pd1[PdI[j]], axis=1)
                Bc[-1][j] = math.factorial(i) / np.prod([math.factorial(p) for p in Pd1[PdI[j]]])

        # Tanh derivatives
        D = tanh_deriv(dv, o + 4)
        print(f"D shape: {D.shape}")
        DD = D[:, 1:] * Ars[:, np.newaxis] - D[:, :-1]
        print(f"DD shape: {DD.shape}")

        # Prefactors
        As = (1 - np.tanh(dv) ** 2) * self.A - np.eye(N)
        AsI = inv(As)
        AsI2 = np.dot(AsI, AsI)
        AsI3 = np.dot(AsI2, AsI)
        AsI4 = np.dot(AsI3, AsI)

        # Sole higher derivative terms
        CM = DD[:, :4].reshape(N, 1, 4)
        for j in range(2, o + 1):
            start_idx = (j - 1) * 4
            end_idx = j * 4
            if end_idx > DD.shape[1]:
                #print(f"Skipping j={j} because end_idx={end_idx} exceeds DD.shape[1]={DD.shape[1]}")
                continue
            DD_segment = DD[:, start_idx:end_idx]
            B_segment = Bc[j - 2] * Bk[j - 2]
            #print(f"DD_segment shape: {DD_segment.shape}")
            #print(f"B_segment shape: {B_segment.shape}")
            CM = np.concatenate(
                (CM, DD_segment.reshape(N, 1, 4) * B_segment[:, np.newaxis] / math.factorial(j - 1)),
                axis=1
            )

        # xdot terms
        C1 = np.dot(AsI, CM[:, :, 0])
        C2 = np.einsum('ij,jkl->ikl', AsI2, CM[:, :, 1] * (Bc[0] * Bk[0])[:, np.newaxis] / gam)
        C3b = C2 / gam
        C4c = C3b / gam

        # xdot^2 terms
        C3a = np.einsum('ij,jkl->ikl', AsI3, CM[:, :, 2] * (Bc[1] * Bk[1])[:, np.newaxis] / gam ** 2)
        C4b = 3 * np.einsum('ij,jkl->ikl', AsI4, CM[:, :, 2] * Bk[1][:, np.newaxis] / gam ** 3)

        # xdot^3 terms
        C4a = np.einsum('ij,jkl->ikl', AsI4, CM[:, :, 3] * (Bc[2] * Bk[2])[:, np.newaxis] / gam ** 3)

        return Pd1, C1, C2, C3a, C3b, C4a, C4b, C4c

def tanh_deriv(d, order):
    """
    Compute the derivatives of the tanh function up to the specified order.
    
    Parameters:
    d (numpy.ndarray): Vector of input values.
    order (int): Order of derivatives to compute.
    
    Returns:
    numpy.ndarray: Matrix of derivatives evaluated at input values.
    """
    z = sp.symbols('z')
    tanh_z = sp.tanh(z)
    derivatives = [tanh_z]
    
    # Compute derivatives symbolically
    for i in range(1, order):
        derivatives.append(sp.diff(derivatives[-1], z))
    
    # Convert symbolic derivatives to numerical functions
    derivative_functions = [sp.lambdify(z, deriv) for deriv in derivatives]
    
    # Evaluate derivatives at input values
    D = np.zeros((len(d), order))
    for i, func in enumerate(derivative_functions):
        D[:, i] = func(d)
    
    return D

