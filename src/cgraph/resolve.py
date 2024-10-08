""" Resolves an CGraph into a Reservoir """

from cgraph.cgraph import CGraph
from prnn.reservoir import Reservoir
import numpy as np
from typing import List
from collections import OrderedDict


class Resolver:
    """
    Resolves an CGraph into a Reservoir
    """

    def __init__(self, cgraph: CGraph, verbose=False):
        self.graph = cgraph
        self.reservoir: Reservoir = None  # map of constituent reservoirs to idx in A
        self.res_idx_map = OrderedDict()
        self.connections_to_clean = set()
        self.verbose = verbose

    def resolve(self) -> Reservoir:
        # TODO: Populate adjacency
        dim = self._get_reservoirs()
        a = self._gen_composite_adjacency(dim)
        a = self._process_connections(a)
        self._cleanup_reservoirs()
        self._combine_reservoirs(a)
        self._remove_ignored_inputs()
        # TODO: check input dims (make list of input names a reservoir member)
        # TODO: restrict output space to ret (make list of output names a reservoir member)

        return self.reservoir

    def _get_reservoirs(self) -> int:
        """
        Retrieves all reservoirs from the CGraph and calculates the total dimension.
        Side effect: updates self.res_idx_map with the indices of each reservoir in the composite.
        Returns:
        * dim: total dimension of all reservoirs combined
        """
        dim = 0
        idx = 0
        for name in self.graph.all_nodes():
            node = self.graph.get_node(name)
            if node["type"] == "reservoir":
                reservoir: Reservoir = node["reservoir"]
                dim += reservoir.A.shape[0]
                self.res_idx_map[reservoir] = idx
                idx += reservoir.A.shape[0]
                if self.verbose:
                    print(reservoir.name)
        return dim

    def _gen_composite_adjacency(self, dim) -> np.ndarray:
        """
        Generates a composite reservoir from the CGraph.
        Returns:
        * a: combined adjacency matrix
        """
        a = np.zeros((dim, dim))
        for r, idx in self.res_idx_map.items():
            sz = r.A.shape[0]
            a[idx : idx + sz, idx : idx + sz] = r.A
        return a

    def _process_connections(self, a):
        """
        Processes the connections in the CGraph. Internalizes connections
        and removes now-internalized rows and columns from W and B.
        """
        removed_indices = {}

        for var in self.graph.all_nodes():
            node = self.graph.get_node(var)
            if node["type"] != "var":
                continue

            # Get source and target reservoirs and indices
            out_res, out_idx = self.graph.get_var_source(var)
            in_res, in_idx = self.graph.get_var_target(var)

            # Adjust indices if columns have been removed
            out_idx = self._adjust_index(out_res, out_idx, removed_indices)
            in_idx = self._adjust_index(in_res, in_idx, removed_indices)

            self.connections_to_clean.add((out_res, out_idx, in_res, in_idx))

            # remove internalized variable for reservoir input/ output lists
            out_res.output_names.remove(var)
            in_res.input_names.remove(var)

            # Get rows and columns from W and B matrices
            try:
                w_row = out_res.W[out_idx - 1, :].reshape(1, -1)
                b_col = in_res.B[:, in_idx - 1].reshape(-1, 1)
            except IndexError as e:
                raise ValueError(f"Index error when processing connection: {e}") from e

            # Compute 'internalized' section of A
            sec = np.outer(b_col, w_row)
            out_pos = self.res_idx_map[out_res]
            in_pos = self.res_idx_map[in_res]
            sec_rows, sec_cols = sec.shape

            # Insert section into comb_a at the correct location
            a[in_pos : in_pos + sec_rows, out_pos : out_pos + sec_cols] += sec

        return a

    def _adjust_index(self, res, idx, removed_indices):
        """
        Adjusts the index based on the number of removed columns.
        """
        if res not in removed_indices:
            removed_indices[res] = []

        for removed_idx in sorted(removed_indices[res]):
            if idx > removed_idx:
                idx -= 1

        return idx

    def _cleanup_reservoirs(self):
        input_indices_to_remove = {}
        output_indices_to_remove = {}

        for out_res, out_idx, in_res, in_idx in self.connections_to_clean:
            out_res: Reservoir
            in_res: Reservoir

            # Collect indices to remove
            if in_res not in input_indices_to_remove:
                input_indices_to_remove[in_res] = []
            if out_res not in output_indices_to_remove:
                output_indices_to_remove[out_res] = []

            input_indices_to_remove[in_res].append(in_idx - 1)
            output_indices_to_remove[out_res].append(out_idx - 1)

        # Remove indices from B and W
        for in_res, indices in input_indices_to_remove.items():
            for idx in sorted(indices, reverse=True):
                in_res.B = (
                    np.delete(in_res.B, idx, axis=1)
                    if in_res.B.shape[1] > 1
                    else np.zeros_like(in_res.B)
                )
                in_res.x_init = (
                    np.delete(in_res.x_init, idx, axis=0)
                    if in_res.x_init.shape[0] > 1
                    else np.zeros((1, 1))
                )

        for out_res, indices in output_indices_to_remove.items():
            for idx in sorted(indices, reverse=True):
                out_res.W = (
                    np.delete(out_res.W, idx, axis=0)
                    if out_res.W.shape[0] > 1
                    else np.zeros((0, out_res.A.shape[0]))
                )

    # check! done by chat
    def _combine_reservoirs(self, a):
        """
        Combines the B and W matrices from multiple reservoirs, placing them
        on their respective diagonals. Also stacks x_init, r_init, and d
        components for each reservoir. Returns a new combined Reservoir.
        """
        # Initialize empty arrays for stacking and diagonal placement
        x_all = np.zeros((0, 1))
        r_all = np.zeros((0, 1))
        d_all = np.zeros((0, 1))
        b_comb = np.zeros((0, 0))
        w_comb = np.zeros((0, 0))
        input_names = []
        output_names = []

        # Track the current row and column indices for placing the next matrix
        b_row, b_col, w_row, w_col = 0, 0, 0, 0

        for res in self.res_idx_map.keys():
            res: Reservoir
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

            # Update input and output names
            for name in res.input_names:
                if name not in input_names:
                    input_names.append(name)

            for name in res.output_names:
                if name not in output_names:
                    output_names.append(name)

        # Create and return a new combined Reservoir
        self.reservoir = Reservoir(
            A=a,
            B=b_comb,
            W=w_comb,
            x_init=x_all,
            r_init=r_all,
            d=d_all,
            global_timescale=0.001,
            gamma=100,
            input_names=input_names,
            output_names=output_names,
        )

        # Remove ignored inputs
        self._remove_ignored_inputs()

    # check! done by chat
    def _remove_ignored_inputs(self):
        """
        Removes columns in B that correspond to zero entries in x_init for the combined reservoir.
        """
        i = 0
        while i < self.reservoir.B.shape[1]:

            # Check if all elements in column i are zeros
            if np.all(self.reservoir.B[:, i] == 0):
                if self.reservoir.B.shape[1] > 1:  # Remove column if not the last one
                    # self.reservoir.input_names.pop(i)

                    self.reservoir.B = np.delete(self.reservoir.B, i, axis=1)
                    self.reservoir.x_init = np.delete(self.reservoir.x_init, i, axis=0)
                else:  # If it's the last column, set to zeros
                    self.reservoir.B = np.zeros_like(self.reservoir.B)
                    self.reservoir.x_init = np.zeros((1, 1))
                    break
            else:
                i += 1
