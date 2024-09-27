""" Circuit class and methods """

import numpy as np
from prnn.reservoir import Reservoir


class Circuit:
    """
    Defines the Circuit type, which describes a graph of connected networks.
    * circuit.connect() links a circuit object into a single reservoir
    """

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

            raise ValueError(
                "Invalid config/reservoirs combinations \
                              passed to connect."
            )

        # circuit components
        dim_a = sum(res.A.shape[0] for res in self.reservoirs)
        comb_a = np.zeros((dim_a, dim_a))

        # put adjacencies on diagonal, note indices
        idx = 0
        reservoir_indices = {}
        for reservoir in self.reservoirs:
            # putsa on diagonal
            a = reservoir.A
            size = a.shape[0]
            reservoir_indices[reservoir] = idx
            comb_a[idx : idx + size, idx : idx + size] = a
            idx += size

        for connection in self.config:
            output_net, a, input_net, b = connection
            output_net: Reservoir
            input_net: Reservoir

            # Internalize connection
            w_row = output_net.W[a - 1, :].reshape(1, -1)  # * r
            b_col = input_net.B[:, b - 1].reshape(-1, 1)
            # a_section = b_col @ w_row
            a_section = np.outer(b_col, w_row)

            # keep track of internalized inputs/ outputs
            output_net.usedOutputs.add(a)  # TODO: make this a set!!
            input_net.usedInputs.add(b)

            # Insert into comb_a at correct position
            output_idx = reservoir_indices[output_net]
            input_idx = reservoir_indices[input_net]
            a_rows, a_cols = a_section.shape
            comb_a[
                input_idx : input_idx + a_rows, output_idx : output_idx + a_cols
            ] += a_section

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
        current_b_row = 0
        current_b_col = 0
        current_w_row = 0
        current_w_col = 0

        for reservoir in self.reservoirs:
            reservoir: Reservoir
            b_matrix = reservoir.B
            w_matrix = reservoir.W
            b_rows, b_cols = b_matrix.shape
            w_rows, w_cols = w_matrix.shape

            # Determine new sizes for B and W
            new_b_row_size = current_b_row + b_rows
            new_b_col_size = current_b_col + b_cols
            new_w_row_size = current_w_row + w_rows
            new_w_col_size = current_w_col + w_cols

            # Resize the B and W arrays to accommodate the new matrices
            if B.size == 0:
                B = np.zeros((new_b_row_size, new_b_col_size))
            else:
                B = np.pad(B, ((0, b_rows), (0, b_cols)), mode="constant")

            if W.size == 0:
                W = np.zeros((new_w_row_size, new_w_col_size))
            else:
                W = np.pad(W, ((0, w_rows), (0, w_cols)), mode="constant")

            # Place the matrices B and W on their respective diagonals
            B[
                current_b_row : current_b_row + b_rows,
                current_b_col : current_b_col + b_cols,
            ] = b_matrix
            W[
                current_w_row : current_w_row + w_rows,
                current_w_col : current_w_col + w_cols,
            ] = w_matrix

            # Update current row and column indices for B and W
            current_b_row += b_rows
            current_b_col += b_cols
            current_w_row += w_rows
            current_w_col += w_cols

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

        return Reservoir(comb_a, B, r_init, x_init, 0.001, 100, d, W)


def validate2reservoirs(config) -> list[Reservoir]:
    reservoirs = set()
    for connection in config:
        if len(connection) != 4:
            raise ValueError(
                "Each connection must have 4 elements: \
                    [reservoir1, output_index, reservoir2, input_index]"
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
