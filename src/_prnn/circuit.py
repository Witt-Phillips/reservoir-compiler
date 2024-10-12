""" Circuit class and methods """

import numpy as np
from _prnn.reservoir import Reservoir


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
        """
        Connects multiple reservoirs based on the specified configuration
        and returns the combined Reservoir object.
        """

        # Validate configuration and reservoirs
        res = self._validate_reservoirs()
        if res:
            return res  # ret of only 1 reservoir

        # put adjacencies of each res in self.reservoirs on diagonal, note indices of each
        a, idxs = self._build_adj()

        # internalize each connection in config into a
        self._process_connections(a, idxs)

        # remove internalized columns of B and W
        for res in self.reservoirs:
            _cleanup_reservoir(res)

        # combine other elements of constituent reservoirs
        b, w, x_init, r_init, d = self._combine_reservoirs()

        # Remove zero columns from B and corresponding entries in x_init
        b, x_init = _remove_zero_columns(b, x_init)

        # TODO: figure out if we only want to return readouts (then map ir readouts to circuit)
        """ 
        if self.readouts is not None:
            for i in range(W.shape[0]):
                if i not in self.readouts:
                    W = np.delete(W, i, axis=0)
        """
        return Reservoir(a, b, r_init, x_init, 0.001, 100, d, w)

    def _build_adj(self):
        """
        Builds the adjacency matrix and tracks reservoir indices.
        Returns:
        * adj: combined adjacency with each res in self.reservoirs
        on adjacency
        * res_idx: index of each constituent reservoir in adj.
        """
        dim = sum(r.A.shape[0] for r in self.reservoirs)
        adj = np.zeros((dim, dim))

        idx = 0
        res_idx = {}
        for r in self.reservoirs:
            sz = r.A.shape[0]
            res_idx[r] = idx
            adj[idx : idx + sz, idx : idx + sz] = r.A
            idx += sz

        return adj, res_idx

    def _process_connections(self, comb_a, res_indices):
        """
        Processes connections between reservoirs and updates the combined adjacency matrix comb_a.
        """
        for out_res, out_idx, in_res, in_idx in self.config:
            out_res: Reservoir
            in_res: Reservoir

            try:
                # Get rows and columns from W and B matrices
                w_row = out_res.W[out_idx - 1, :].reshape(1, -1)
                b_col = in_res.B[:, in_idx - 1].reshape(-1, 1)
            except IndexError as e:
                raise ValueError(f"Index error when processing connection: {e}") from e

            # Compute the outer product to internalize the connection
            sec = np.outer(b_col, w_row)

            # Track used inputs and outputs
            out_res.usedOutputs.add(out_idx)
            in_res.usedInputs.add(in_idx)

            # Retrieve index positions and validate dimensions before insertion
            try:
                out_pos = res_indices[out_res]
                in_pos = res_indices[in_res]
                sec_rows, sec_cols = sec.shape

                if (
                    comb_a.shape[0] < in_pos + sec_rows
                    or comb_a.shape[1] < out_pos + sec_cols
                ):
                    raise ValueError(
                        "Combined adjacency matrix size mismatch with reservoirs"
                    )

                # Insert section into comb_a at the correct location
                comb_a[in_pos : in_pos + sec_rows, out_pos : out_pos + sec_cols] += sec
            except KeyError as e:
                raise ValueError(
                    f"Reservoir not found in reservoir indices: {e}"
                ) from e

    def _combine_reservoirs(self):
        """
        Combines the B and W matrices from multiple reservoirs, placing them
        on their respective diagonals. Also stacks x_init, r_init, and d
        components for each reservoir.
        """

        # Initialize empty arrays for stacking and diagonal placement
        x_all = np.zeros((0, 1))
        r_all = np.zeros((0, 1))
        d_all = np.zeros((0, 1))
        b_comb = np.zeros((0, 0))
        w_comb = np.zeros((0, 0))

        # Track the current row and column indices for placing the next matrix
        b_row, b_col, w_row, w_col = 0, 0, 0, 0

        for res in self.reservoirs:
            b, w = res.B, res.W
            b_r, b_c = b.shape
            w_r, w_c = w.shape

            # Resize B and W arrays to fit the new matrices
            b_comb = (
                np.pad(b_comb, ((0, b_r), (0, b_c)), mode="constant")
                if b_comb.size
                else np.zeros((b_row + b_r, b_col + b_c))
            )
            w_comb = (
                np.pad(w_comb, ((0, w_r), (0, w_c)), mode="constant")
                if w_comb.size
                else np.zeros((w_row + w_r, w_col + w_c))
            )

            # Place the matrices B and W on their respective diagonals
            b_comb[b_row : b_row + b_r, b_col : b_col + b_c] = b
            w_comb[w_row : w_row + w_r, w_col : w_col + w_c] = w

            # Update row and column indices
            b_row, b_col = b_row + b_r, b_col + b_c
            w_row, w_col = w_row + w_r, w_col + w_c

            # Stack x_init, r_init, and d
            x_all = np.vstack([x_all, res.x_init])
            r_all = np.vstack([r_all, res.r_init])
            d_all = np.vstack([d_all, res.d])

        return b_comb, w_comb, x_all, r_all, d_all

    def _validate_reservoirs(self):
        """
        Validate reservoirs and configuration.
        If there are no connections, return the first reservoir if only one exists.
        Returns None if more than one reservoir or a configuration exists.
        """
        if not self.config and len(self.reservoirs) == 1:
            return next(iter(self.reservoirs))
        if not self.config:
            raise ValueError(
                "Invalid config/reservoirs combinations passed to connect."
            )
        return None


def _remove_zero_columns(b, x_init):
    """
    Removes columns in B that correspond to zero entries in x_init.

    Args:
        B (np.ndarray): Weight matrix.
        x_init (np.ndarray): Input vector.

    Returns:
        B (np.ndarray): Updated B matrix.
        x_init (np.ndarray): Updated x_init vector.
    """
    i = 0
    while i < b.shape[1]:
        # Check if all elements in column i are zeros
        if np.all(b[:, i] == 0):
            if b.shape[1] > 1:  # Remove column if not the last one
                b = np.delete(b, i, axis=1)
                x_init = np.delete(x_init, i, axis=0)
            else:  # If it's the last column, set to zeros
                b = np.zeros_like(b)
                x_init = np.zeros((1, 1))
                break
        else:
            i += 1
    return b, x_init


def _cleanup_reservoir(reservoir: Reservoir):
    """Remove used inputs from B and x_init, and used outputs from W for a reservoir."""

    # Remove used inputs from B and x_init
    for inp in reservoir.usedInputs:
        reservoir.B = (
            np.delete(reservoir.B, inp - 1, axis=1)
            if reservoir.B.shape[1] > 1
            else np.zeros_like(reservoir.B)
        )
        reservoir.x_init = (
            np.delete(reservoir.x_init, inp - 1, axis=0)
            if reservoir.x_init.shape[0] > 1
            else np.zeros((1, 1))
        )

    # Remove used outputs from W
    for out in reservoir.usedOutputs:
        reservoir.W = (
            np.delete(reservoir.W, out - 1, axis=0)
            if reservoir.W.shape[0] > 1
            else np.zeros((0, reservoir.A.shape[0]))
        )


def validate2reservoirs(config) -> list[Reservoir]:
    """
    Validates connections (config) of a circuit object
    """
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
