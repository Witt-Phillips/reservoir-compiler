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