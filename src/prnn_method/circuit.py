import numpy as np
from reservoir import Reservoir

# can't pass duplicate reservoirs (yet, at least)
def connect(circuit, reservoirs) -> Reservoir:
    # circuit components
    dimA = sum(res.A.shape[0] for res in reservoirs)
    combA = np.zeros((dimA, dimA)) #TODO combA

    # put adjacencies on diagonal, note indices
    idx = 0
    reservoir_indices = {}
    for reservoir in reservoirs:
        #puts A on diagonal
        A = reservoir.A
        size = A.shape[0]
        reservoir_indices[reservoir] = idx
        combA[idx:idx+size, idx:idx+size] = A
        idx += size

    # code circuit connections
    #TODO: calculate reservoir list during circuit validation instead of passing in
    #reservoirs = set()

    for connection in circuit:
        outputNet, a, inputNet, b = connection
        #reservoirs.add(outputNet)
        #reservoirs.add(inputNet)

        # Internalize connection
        W_row = outputNet.W[a - 1, :].reshape(1, -1) # * r
        B_col = inputNet.B[:, b - 1].reshape(-1, 1)
        #A_section = B_col @ W_row
        A_section = np.outer(B_col, W_row)

        # keep track of internalized inputs/ outputs
        outputNet.usedOutputs.append(a)
        inputNet.usedInputs.append(b)

        # Insert into combA at correct position
        output_idx = reservoir_indices[outputNet]
        input_idx = reservoir_indices[inputNet]
        A_rows, A_cols = A_section.shape
        combA[input_idx:input_idx+A_rows, output_idx:output_idx+A_cols] += A_section
    
    
    for reservoir in reservoirs:
        # delete used inputs
        for folded_input in reservoir.usedInputs:
            if reservoir.B.shape[1] > 1:
                reservoir.B = np.delete(reservoir.B, folded_input - 1, axis=1)
            else:
                reservoir.B = np.zeros_like(reservoir.B)

            if reservoir.x_init.shape[0] > 1:
                reservoir.x_init = np.delete(reservoir.x_init, folded_input - 1, axis=0)
            else:
                reservoir.x_init = np.zeros((1, 1))

        # delete used rows of W (outputs)
        for folded_output in reservoir.usedOutputs:
            if reservoir.W.shape[0] > 1:
                reservoir.W = np.delete(reservoir.W, folded_output - 1, axis=0)
            else:
                reservoir.W = np.zeros_like(reservoir.W)

    # Initialize empty arrays for stacking and diagonal placement
    x_init = np.array([]).reshape(0, 1)
    r_init = np.array([]).reshape(0, 1)
    d = np.array([]).reshape(0, 1)
    B = np.array([]).reshape(0, 0)
    W = np.array([]).reshape(0, 0)

    # Track the current row and column indices for placing the next matrix
    current_B_row = 0
    current_B_col = 0
    current_W_row = 0
    current_W_col = 0

    for reservoir in reservoirs:
        reservoir: Reservoir
        B_matrix = reservoir.B
        W_matrix = reservoir.W
        B_rows, B_cols = B_matrix.shape
        W_rows, W_cols = W_matrix.shape

        # Determine new sizes for B and W
        new_B_row_size = current_B_row + B_rows
        new_B_col_size = current_B_col + B_cols
        new_W_row_size = current_W_row + W_rows
        new_W_col_size = current_W_col + W_cols

        # Resize the B and W arrays to accommodate the new matrices
        if B.size == 0:
            B = np.zeros((new_B_row_size, new_B_col_size))
        else:
            B = np.pad(B, ((0, B_rows), (0, B_cols)), mode='constant')
        
        if W.size == 0:
            W = np.zeros((new_W_row_size, new_W_col_size))
        else:
            W = np.pad(W, ((0, W_rows), (0, W_cols)), mode='constant')

        # Place the matrices B and W on their respective diagonals
        B[current_B_row:current_B_row+B_rows, current_B_col:current_B_col+B_cols] = B_matrix
        W[current_W_row:current_W_row+W_rows, current_W_col:current_W_col+W_cols] = W_matrix

        # Update current row and column indices for B and W
        current_B_row += B_rows
        current_B_col += B_cols
        current_W_row += W_rows
        current_W_col += W_cols

        # Stack the other components
        x_init = np.vstack([x_init, reservoir.x_init])
        r_init = np.vstack([r_init, reservoir.r_init])
        d = np.vstack([d, reservoir.d])

    #TODO: remove superflous (all 0) cols of B/ x_inits.


    return Reservoir(combA, B, r_init, x_init, .001, 100, d, W)
    
