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
        a = self._gen_composite_adjacency()
        a = self._process_connections(a)
        self._cleanup_reservoirs()
        self._combine_reservoirs(a)
        self._remove_ignored_inputs()
        # TODO: check input dims (make list of input names a reservoir member)
        # TODO: restrict output space to ret (make list of output names a reservoir member)

        return self.reservoir

    def _gen_composite_adjacency(self) -> np.ndarray:
        """
        Generates a composite reservoir from the CGraph.
        Returns:
        * a: combined adjacency matrix
        side effect: updates self.res_idx_map with the indices of
        each reservoir in the composite
        """

        # pass 1: get all reservoirs and their dimensions
        reservoirs: List[Reservoir] = []
        dim = 0
        for name in self.graph.all_nodes():
            node = self.graph.get_node(name)
            if node["type"] == "reservoir":
                reservoir: Reservoir = node["reservoir"]
                dim += reservoir.A.shape[0]
                reservoirs.append(reservoir)
                if self.verbose:
                    print(reservoir.name)
        if self.verbose:
            print("Dim composite A:", dim)

        a = np.zeros((dim, dim))
        idx = 0
        for r in reservoirs:
            sz = r.A.shape[0]
            self.res_idx_map[r] = idx
            a[idx : idx + sz, idx : idx + sz] = r.A
            idx += sz

        return a

    def _process_connections(self, a):
        """
        Processes the connections in the CGraph. Interalizes connections
        and removes now-internalized rows and columns from W and B.
        """
        for var in self.graph.all_nodes():
            node = self.graph.get_node(var)
            if node["type"] != "var":
                continue

            out_res, out_idx = self.graph.get_var_source(var)
            in_res, in_idx = self.graph.get_var_target(var)
            self.connections_to_clean.add((out_res, out_idx, in_res, in_idx))

            if self.verbose:
                print(f"Processing connection: {var} -> {out_res} -> {in_res}")

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

    def _cleanup_reservoirs(self):
        for out_res, out_idx, in_res, in_idx in self.connections_to_clean:
            out_res: Reservoir
            in_res: Reservoir
            # Remove used inputs from B and x_init for in_res
            in_res.B = (
                np.delete(in_res.B, in_idx - 1, axis=1)
                if in_res.B.shape[1] > 1
                else np.zeros_like(in_res.B)
            )
            in_res.x_init = (
                np.delete(in_res.x_init, in_idx - 1, axis=0)
                if in_res.x_init.shape[0] > 1
                else np.zeros((1, 1))
            )

            # Remove used outputs from W for out_res
            out_res.W = (
                np.delete(out_res.W, out_idx - 1, axis=0)
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
                    self.reservoir.B = np.delete(self.reservoir.B, i, axis=1)
                    self.reservoir.x_init = np.delete(self.reservoir.x_init, i, axis=0)
                else:  # If it's the last column, set to zeros
                    self.reservoir.B = np.zeros_like(self.reservoir.B)
                    self.reservoir.x_init = np.zeros((1, 1))
                    break
            else:
                i += 1
