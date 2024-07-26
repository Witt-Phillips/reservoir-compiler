import numpy as np

def connect(circuit, reservoirs):
    # init
    dimA = sum(res.A.shape[0] for res in reservoirs)
    compA = np.zeros((dimA, dimA))

    # put adjacencies on diagonal, note indices
    idx = 0
    reservoir_indices = {}
    for reservoir in reservoirs:
        A = reservoir.A
        size = A.shape[0]
        reservoir_indices[reservoir] = idx
        compA[idx:idx+size, idx:idx+size] = A
        idx += size

    # Code circuit connections
    for connection in circuit:
        outputNet, o, inputNet, x = connection

        # BW
        w_slice = outputNet.W[o - 1, :].reshape(1, -1)
        b_slice = outputNet.B[:, x - 1].reshape(-1, 1)
        BW = b_slice @ w_slice

        # Insert into compA at correct position
        output_idx = reservoir_indices[outputNet]
        input_idx = reservoir_indices[inputNet]
        BW_rows, BW_cols = BW.shape
        compA[input_idx:input_idx+BW_rows, output_idx:output_idx+BW_cols] += BW
    np.set_printoptions(precision=2, linewidth=1000, suppress=True)
    print(compA)
    return compA