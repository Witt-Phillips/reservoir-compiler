import numpy as np
from prnn.reservoir import Reservoir


class Circuit:
    def __init__(
        self, config, readouts=None, preserve_reservoirs=False, reservoirs=None
    ):
        self.config = config
        self.reservoirs = (
            reservoirs if reservoirs is not None else validate2reservoirs(config)
        )
        self.readouts = readouts

        # copy reservoirs to avoid overwrites
        if preserve_reservoirs:
            new_reservoirs = []
            for reservoir in self.reservoirs:
                copied_res = reservoir.copy()
                new_reservoirs.append(copied_res)

                for connection in config:
                    if connection[0] == reservoir:
                        connection[0] = copied_res
                    if connection[2] == reservoir:
                        connection[2] = copied_res
            self.reservoirs = new_reservoirs

    def connect(self) -> Reservoir:
        # If there are reservoirs, return the first one
        if not self.config:
            l = len(self.reservoirs)
            if l == 1:
                return next(iter(self.reservoirs))
            elif l > 1:
                raise ValueError(
                    "Multiple reservoirs exist but config is empty. Cannot determine which reservoir to return."
                )
            else:
                raise ValueError("No reservoirs available to connect.")

        # circuit components
        dimA = sum(res.A.shape[0] for res in self.reservoirs)
        combA = np.zeros((dimA, dimA))  # TODO combA

        # put adjacencies on diagonal, note indices
        idx = 0
        reservoir_indices = {}
        for reservoir in self.reservoirs:
            # puts A on diagonal
            A = reservoir.A
            size = A.shape[0]
            reservoir_indices[reservoir] = idx
            combA[idx : idx + size, idx : idx + size] = A
            idx += size

        for connection in self.config:
            outputNet, a, inputNet, b = connection
            outputNet: Reservoir
            inputNet: Reservoir

            # Internalize connection
            W_row = outputNet.W[a - 1, :].reshape(1, -1)  # * r
            B_col = inputNet.B[:, b - 1].reshape(-1, 1)
            # A_section = B_col @ W_row
            A_section = np.outer(B_col, W_row)

            # keep track of internalized inputs/ outputs
            outputNet.usedOutputs.add(a)  # TODO: make this a set!!
            inputNet.usedInputs.add(b)

            # Insert into combA at correct position
            output_idx = reservoir_indices[outputNet]
            input_idx = reservoir_indices[inputNet]
            A_rows, A_cols = A_section.shape
            combA[
                input_idx : input_idx + A_rows, output_idx : output_idx + A_cols
            ] += A_section

        for reservoir in self.reservoirs:
            # delete used inputs
            for folded_input in reservoir.usedInputs:
                if reservoir.B.shape[1] > 1:
                    reservoir.B = np.delete(reservoir.B, folded_input - 1, axis=1)
                else:
                    reservoir.B = np.zeros_like(reservoir.B)

                if reservoir.x_init.shape[0] > 1:
                    reservoir.x_init = np.delete(
                        reservoir.x_init, folded_input - 1, axis=0
                    )
                else:
                    reservoir.x_init = np.zeros((1, 1))

            # delete used rows of W (outputs)
            for folded_output in reservoir.usedOutputs:
                if reservoir.W.shape[0] > 1:
                    reservoir.W = np.delete(reservoir.W, folded_output - 1, axis=0)
                else:
                    # reservoir.W = np.zeros_like(reservoir.W)
                    reservoir.W = np.zeros((0, reservoir.A.shape[0]))

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

        for reservoir in self.reservoirs:
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
                B = np.pad(B, ((0, B_rows), (0, B_cols)), mode="constant")

            if W.size == 0:
                W = np.zeros((new_W_row_size, new_W_col_size))
            else:
                W = np.pad(W, ((0, W_rows), (0, W_cols)), mode="constant")

            # Place the matrices B and W on their respective diagonals
            B[
                current_B_row : current_B_row + B_rows,
                current_B_col : current_B_col + B_cols,
            ] = B_matrix
            W[
                current_W_row : current_W_row + W_rows,
                current_W_col : current_W_col + W_cols,
            ] = W_matrix

            # Update current row and column indices for B and W
            current_B_row += B_rows
            current_B_col += B_cols
            current_W_row += W_rows
            current_W_col += W_cols

            # Stack the other components
            x_init = np.vstack([x_init, reservoir.x_init])
            r_init = np.vstack([r_init, reservoir.r_init])
            d = np.vstack([d, reservoir.d])

        # TODO: remove superflous (all 0) cols of B/ x_inits.
        finished = False
        i = 0
        while finished == False:
            B, x_init, removed, finished = purge_Bx(B, x_init, i)
            if not removed:
                i = i + 1

        # if self.readouts is not None:
        #     for i in range(W.shape[0]):
        #         if i not in self.readouts:
        #             W = np.delete(W, i, axis=0)

        return Reservoir(combA, B, r_init, x_init, 0.001, 100, d, W)


def validate2reservoirs(config) -> list[Reservoir]:
    reservoirs = set()
    for connection in config:
        if len(connection) != 4:
            raise ValueError(
                "Each connection must have 4 elements: [reservoir1, output_index, reservoir2, input_index]"
            )
        if not isinstance(connection[0], Reservoir):
            raise ValueError(
                "Each connection must have a reservoir as the first element"
            )
        if not isinstance(connection[2], Reservoir):
            raise ValueError(
                "Each connection must have a reservoir as the third element"
            )
        if not isinstance(connection[1], int):
            raise ValueError(
                "Each connection must have an integer as the second element"
            )
        if not isinstance(connection[3], int):
            raise ValueError(
                "Each connection must have an integer as the fourth element"
            )

        reservoirs.add(connection[0])
        reservoirs.add(connection[2])

    return reservoirs


def purge_Bx(B, x, i):
    flag = False
    removed_col = False
    finished = False
    # check if all 0s
    for j in range(B.shape[0]):
        if B[j, i] != 0:
            flag = True

    # remove col if all zeros & isn't last col
    if not flag:
        if B.shape[1] > 1:
            B = np.delete(B, i, axis=1)  # delete row in W
            x = np.delete(x, i, axis=0)  # delete corresponding input
            removed_col = True

    # are we looking at the last column in B? Base case
    if i == (B.shape[1] - 1):
        finished = True

    return B, x, removed_col, finished
