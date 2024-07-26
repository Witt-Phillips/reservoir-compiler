import numpy as np
from reservoir import Reservoir

# can't pass duplicate reservoirs (yet, at least)
def connect(circuit, reservoirs):
    # circuit components
    dimA = sum(res.A.shape[0] for res in reservoirs)
    compA = np.zeros((dimA, dimA))
    r_inits = np.vstack([res.r_init for res in reservoirs])
    ds = np.vstack([res.d for res in reservoirs])

    # put adjacencies on diagonal, note indices
    idx = 0
    reservoir_indices = {}
    for reservoir in reservoirs:
        A = reservoir.A
        size = A.shape[0]
        reservoir_indices[reservoir] = idx
        compA[idx:idx+size, idx:idx+size] = A
        idx += size

    # code circuit connections
    for connection in circuit:
        outputNet, o, inputNet, x = connection

        # BW
        w_slice = outputNet.W[o - 1, :].reshape(1, -1)
        b_slice = outputNet.B[:, x - 1].reshape(-1, 1)
        BW = b_slice @ w_slice
        
        # TODO: remove slice logic
        
        # Insert into compA at correct position
        output_idx = reservoir_indices[outputNet]
        input_idx = reservoir_indices[inputNet]
        BW_rows, BW_cols = BW.shape
        compA[input_idx:input_idx+BW_rows, output_idx:output_idx+BW_cols] += BW
    
    # TODO: figure out B logic (when we have inputs)
    OB = np.zeros((dimA, 1))
    x_init = np.zeros((1, 1))

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(compA)

    # TODO: figure out W logic


    #reservoir = Reservoir(compA, OB, r_inits, x_init, .1, 100, ds, )
    
