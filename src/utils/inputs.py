import numpy as np

def high_low_inputs(time):
    ot = np.ones((2, time, 4))
    pt = np.concatenate((
        np.array([[-0.1], [-0.1]])[:, np.newaxis] * ot,
        np.array([[-0.1], [0.1]])[:, np.newaxis] * ot,
        np.array([[0.1], [-0.1]])[:, np.newaxis] * ot,
        np.array([[0.1], [0.1]])[:, np.newaxis] * ot),
        axis=1)
    return pt

def zeros(time):
    return np.zeros((1, time, 4))

def sr_inputs(time):
    ot = np.ones((2, time, 4))
    pt = np.concatenate((
        np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot[:, :1000, :],
        np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot,
        np.array([[1], [-1]])[::-1, :, np.newaxis] * ot[:, :500, :],
        np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot,
        np.array([[-1], [1]])[::-1, :, np.newaxis] * ot[:, :500, :],
        np.array([[-1], [-1]])[::-1, :, np.newaxis] * ot
    ), axis=1) * 0.1

    return pt


# manually generated lorenz attractor
def lorenz(time, dt=0.01):
    time -= 1
    x, y, z = np.zeros((3, time + 1, 4))
    x[0, :], y[0, :], z[0, :] = 0, 1, 1.05

    for i in range(time):
        dx, dy, dz = lorenz_engine(x[i], y[i], z[i])
        x[i + 1, :] = x[i, :] + dx * dt
        y[i + 1, :] = y[i, :] + dy * dt
        z[i + 1, :] = z[i, :] + dz * dt

    outputs = np.stack((x, y, z), axis=0)
    print(outputs.shape)
    return outputs

def lorenz_engine(x, y, z, sig=10, rho=28, beta=8/3):
    dx = sig * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz
