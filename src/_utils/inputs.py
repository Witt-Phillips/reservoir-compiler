""" 
Util functions for generating inputs to RNNs
"""

import numpy as np


def high_low_inputs(time):
    """Creates every high low permutation for two inputs"""
    base_patterns = np.array([[-0.1, 0.1, -0.1, 0.1], [-0.1, -0.1, 0.1, 0.1]])

    reps = time // 4
    row1 = np.repeat(base_patterns[0], reps)
    row2 = np.repeat(base_patterns[1], reps)

    # ensure # cols = time
    if len(row1) < time:
        extra = time - len(row1)
        row1 = np.concatenate((row1, np.tile(base_patterns[0], extra)[:extra]))
        row2 = np.concatenate((row2, np.tile(base_patterns[1], extra)[:extra]))

    return np.vstack((row1, row2))


def high_low_inputs_3rows(time):
    """High low perms for 3 inputs"""
    base_patterns = np.array([[-0.1, 0.1, -0.1, 0.1], [-0.1, -0.1, 0.1, 0.1]])

    reps = time // 4
    row1 = np.repeat(base_patterns[0], reps)
    row2 = np.repeat(base_patterns[1], reps)

    # ensure the number of columns = time
    if len(row1) < time:
        extra = time - len(row1)
        row1 = np.concatenate((row1, np.tile(base_patterns[0], extra)[:extra]))
        row2 = np.concatenate((row2, np.tile(base_patterns[1], extra)[:extra]))

    # Copy the second row as the third row
    row3 = np.copy(row2)

    # Stack all three rows
    return np.vstack((row1, row2, row3))


def zeros(time):
    """Wrapper on np.zeros"""
    return np.zeros((1, time))


def sr_inputs(time):
    """Inputs for an SR-LATCH"""
    ot = np.ones((2, time, 4))
    pt = (
        np.concatenate(
            (
                np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot[:, :1000, :],
                np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot,
                np.array([[1], [-1]])[::-1, :, np.newaxis] * ot[:, :500, :],
                np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot,
                np.array([[-1], [1]])[::-1, :, np.newaxis] * ot[:, :500, :],
                np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot,
            ),
            axis=1,
        )
        * 0.1
    )

    return pt[:, :, 0]


# manually generated lorenz attractor
def lorenz(time, dt=0.01):
    """Inputs for a lorenz series"""
    time -= 1
    x, y, z = np.zeros((3, time + 1))
    x[0], y[0], z[0] = 0, 1, 1.05

    for i in range(time):
        dx, dy, dz = lorenz_engine(x[i], y[i], z[i])
        x[i + 1] = x[i] + dx * dt
        y[i + 1] = y[i] + dy * dt
        z[i + 1] = z[i] + dz * dt

    return np.stack((x, y, z), axis=0)


def lorenz_engine(x, y, z):
    """Manually generates a lorenz series"""
    sig = 10
    rho = 28
    beta = 8 / 3
    dx = sig * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz
