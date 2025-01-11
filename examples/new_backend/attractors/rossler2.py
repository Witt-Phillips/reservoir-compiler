from pyres import Reservoir
import numpy as np
import matplotlib.pyplot as plt

# initialize empty reservoir
R: Reservoir = Reservoir(
    np.zeros((3, 3)), np.zeros((3, 2)), np.zeros((3, 1)), np.zeros((2, 1))
)

# rossler eqs (lambda)
eqs = lambda x: [
    -5 * x[1] - 5 * x[2] - 5 * 3 / 5,  # Equation for o1
    5 * x[0] + x[1],  # Equation for o2
    (50 * 5 / 3 * x[0]) * x[2]
    + 50 * x[0]
    - 28.5 * x[2]
    - 28.4 * 3 / 5,  # Equation for o3
]

# solve for lambda w/ identity mapped recurrence
R = R.gen_baseRNN(eqs, 3, 3, set(((0, 0), (1, 1), (2, 2))))

# run fwd for no input.
rL = np.zeros((R.A.shape[0], 30000))
R.r = R.r_init
for i in range(rL.shape[1]):
    xp = np.repeat(R.W @ R.r, 4, 1)
    rL[:, i : i + 1] = R.propagate(xp)

xL = R.W @ rL
ax = plt.figure().add_subplot(projection="3d")
ax.plot(xL[0, :], xL[1, :], xL[2, :])
print(np.linalg.norm(R.W))
plt.show()
