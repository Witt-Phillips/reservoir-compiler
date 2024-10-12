""" Resolves an CGraph into a Reservoir """

from typing import List
from collections import OrderedDict
import numpy as np
from _cgraph.cgraph import CGraph
from _prnn.reservoir import Reservoir


class Resolver:
    """
    Resolves an CGraph into a Reservoir
    """

    def __init__(self, cgraph: CGraph, verbose=False):
        self.graph = cgraph
        self.reservoir: Reservoir = None  # map of constituent reservoirs to idx in A
        self.res_idx_map = OrderedDict()
        self.verbose = verbose

    def resolve(self) -> Reservoir:
        self._track_inp_out()
        dim = self._get_reservoirs()
        a = self._gen_composite_adjacency(dim)
        a = self._process_connections(a)
        self._combine_reservoirs(a)
        self._remove_ignored_inputs()
        self._internalize_constant_inputs()
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
        return dim

    def _track_inp_out(self):
        # Initialize input_names and output_names for each reservoir
        for node_name in self.graph.all_nodes():
            node = self.graph.get_node(node_name)
            if node["type"] == "reservoir":
                node["reservoir"].output_names = []
                node["reservoir"].input_names = []

        # Iterate through each connection (edge) in the graph
        for src, dst in self.graph.graph.edges:
            src_node = self.graph.get_node(src)
            dst_node = self.graph.get_node(dst)

            # If src is a reservoir, add dst to its outputs
            if src_node["type"] == "reservoir":
                src_node["reservoir"].output_names.append(dst)

            # If dest is a reservoir, add src to its inputs
            if dst_node["type"] == "reservoir":
                dst_node["reservoir"].input_names.append(src)

    def _gen_composite_adjacency(self, dim) -> np.ndarray:
        """
        Generates a composite reservoir from the CGraph.
        Returns:
        * a: combined adjacency matrix
        """
        a = np.zeros((dim, dim))
        for r, idx in self.res_idx_map.items():
            r: Reservoir
            sz = r.A.shape[0]
            a[idx : idx + sz, idx : idx + sz] = r.A
        return a

    def _process_connections(self, a):
        for var in self.graph.all_nodes():
            node = self.graph.get_node(var)
            if node["type"] != "var":
                continue

            src_res, _ = self.graph.get_var_source(var)
            tar_res, _ = self.graph.get_var_target(var)
            src_res_idx = src_res.output_names.index(var)
            tar_res_idx = tar_res.input_names.index(var)

            # Internalized section of A
            w_row = src_res.W[src_res_idx, :].reshape(1, -1)
            b_col = tar_res.B[:, tar_res_idx].reshape(-1, 1)
            sec = np.outer(b_col, w_row)
            out_pos = self.res_idx_map[src_res]
            in_pos = self.res_idx_map[tar_res]
            sec_rows, sec_cols = sec.shape
            a[in_pos : in_pos + sec_rows, out_pos : out_pos + sec_cols] += sec

            # cleanup removed connection
            self._remove_res_input(tar_res, tar_res_idx)
            self._remove_res_output(src_res, src_res_idx)

            # cleanup removed connection
            src_res.output_names.remove(var)
            tar_res.input_names.remove(var)

        return a

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
                input_names.append(name)

            for name in res.output_names:
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

    def _remove_ignored_inputs(self):
        """
        Removes columns in B that correspond to zero entries in x_init for the combined reservoir.
        """
        for i in reversed(range(self.reservoir.B.shape[1])):
            # Check if all elements in column i are zeros
            if np.all(self.reservoir.B[:, i] == 0):
                self._remove_res_input(self.reservoir, i)

    def _internalize_constant_inputs(self):
        # we reverse so we pop cols from the end to avoid index shifting
        inp_names = self.reservoir.input_names.copy()
        for idx, inp_name in reversed(list(enumerate(inp_names))):
            node = self.graph.get_node(inp_name)
            if node["value"] is not None:
                b_col = self.reservoir.B[:, idx].reshape(-1, 1)
                x_row = self.reservoir.x_init[idx, :].reshape(1, -1)
                self.reservoir.d += b_col @ x_row
                self._remove_res_input(self.reservoir, idx)

    @staticmethod
    def _remove_res_input(res: Reservoir, idx: int):
        if res.B.shape[1] > 1:
            res.B = np.delete(res.B, idx, axis=1)
        else:
            res.B = np.zeros_like(res.B)

        if res.x_init.shape[0] > 1:
            res.x_init = np.delete(res.x_init, idx, axis=0)
        else:
            res.x_init = np.zeros((1, res.x_init.shape[1]))

    @staticmethod
    def _remove_res_output(res: Reservoir, idx: int):
        res.W = (
            np.delete(res.W, idx, axis=0)
            if res.W.shape[0] > 1
            else np.zeros((0, res.A.shape[0]))
        )
