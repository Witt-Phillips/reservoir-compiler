from typing import Optional, Union, List, Tuple
import numbers
import numpy as np
from ir.lang import Prog, Expr, Opc, Operand
from cgraph.cgraph import CGraph
from ir.fn_library import rnn_lib
from prnn.reservoir import Reservoir


class Core:
    """
    Implements the core of the reservoir language compiler, compiling IR expressions -> networkx graph.
    """

    uid = 0

    def __init__(self, prog: Prog, funcs={}, verbose=False):
        self.graph = CGraph()
        self.vars = set()  # set of declared symbols
        self.inps = set()
        self.prog = prog
        self.funcs: dict[str, Tuple[int, int, Reservoir]] = funcs
        self.verbose = verbose

    def compile_to_cgraph(self) -> CGraph:
        if self.verbose:
            print("Starting compilation of program:")
            print(self.prog)

        for expr in self.prog.exprs:
            if self.verbose:
                print(f"Processing expression: {expr}")
            self._process_expr(expr)

        return self.graph

    def _process_expr(self, expr: Expr) -> Union[None, Reservoir]:
        if self.verbose:
            print(f"Processing opcode: {expr.op}")
        match expr.op:
            case Opc.LET:
                return self._handle_let(expr.operands)
            case Opc.INPUT:
                return self._handle_input(expr.operands[0])
            case Opc.RET:
                return self._handle_ret(expr.operands[0])
            case _:
                return self._handle_custom_opcode(expr)

    def _handle_let(self, operands: List[Operand]) -> None:
        #  Declaration without definition
        if len(operands) == 1:
            for name in operands[0]:
                self.graph.add_var(name)
            return

        # Bind float value to input
        if isinstance(operands[1], numbers.Number):
            val = float(operands[1])
            name = operands[0][0]

            assert (
                name not in self.vars
            ), f"Attempted to bind value to variable: {name} designated as reservoir output."

            if name not in self.inps:
                self.inps.add(name)
                self.graph.add_input(name, val=val)
            else:
                node = self.graph.get_node(name)
                node["value"] = val

            if self.verbose:
                print(f"Set input {name} to {val}")
            return

        # Bind reservoir to set of vars
        names, value = operands
        if self.verbose:
            print(f"LET expression with variables {names} and value {value}")
        # Strictest version, assume process_operand always returns a reservoir
        res: Reservoir = self._process_expr(value)

        if self.verbose:
            print(f"Processed LET value, resulting in reservoir: {res.name}")

        for i, name in enumerate(names):
            if self.verbose:
                print(
                    f"Binding variable {name} to reservoir output {res.name}, index {i}"
                )
            self.graph.add_var(name)
            self.graph.add_edge(res.name, name, out_idx=i)

    def _handle_input(self, operand: Operand) -> None:
        if self.verbose:
            print(f"Handling INPUT operand: {operand}")

        for name in operand:
            self.inps.add(name)
            self.graph.add_input(name)
            if self.verbose:
                print(f"Declared input variable: {name}")

    def _handle_ret(self, operand: Operand) -> None:
        if self.verbose:
            print(f"Handling INPUT operand: {operand}")
        for name in operand:
            self.graph.make_return(name)
            if self.verbose:
                print(f"Converted to return variable: {name}")

    def _handle_custom_opcode(self, expr: Expr) -> Reservoir:
        opcode = expr.op
        assert isinstance(opcode, str), "Custom opcode must be a string"

        _, _, res = self._res_from_lib(opcode)
        self.graph.add_reservoir(res.name, reservoir=res)
        # Check operands
        operands = expr.operands
        for i, sym in enumerate(operands):
            if self.verbose:
                print(f"Processing operand {i}: {sym}")
            if (sym not in self.inps) and (sym not in self.vars):
                ValueError(f"Used undefined symbol {sym}")

            self.graph.add_edge(sym, res.name, in_idx=i)
            if self.verbose:
                print(
                    f"Connected operand {sym} to reservoir {res.name} at input index {i}"
                )
        return res

    def _generate_uid(self) -> str:
        """
        Generates a unique ID for reservoirs.
        """
        Core.uid += 1
        uid = f"res_{Core.uid}"
        if self.verbose:
            print(f"Generated unique reservoir ID: {uid}")
        return uid

    def _res_from_lib(self, opcode: str) -> Tuple[int, int, Reservoir]:
        """
        Retrieves a reservoir from the library based on the given opcode.
        Searches both rnn_lib (which gets preference) and then self.funcs.
        """
        if opcode in rnn_lib:
            if self.verbose:
                print(f"Found opcode {opcode} in rnn_lib")
            inp_dim, out_dim, res_path = rnn_lib[opcode]
            res = Reservoir.load(res_path).copy()
            res.name = self._generate_uid()
            if self.verbose:
                print(
                    f"Loaded reservoir from path {res_path}, assigned name {res.name}"
                )
            return inp_dim, out_dim, res
        elif opcode in self.funcs:
            if self.verbose:
                print(f"Found opcode {opcode} in self.funcs")
            inp_dim, out_dim, res = self.funcs[opcode]
            res = res.copy()
            res.name = self._generate_uid()
            if self.verbose:
                print(f"Created reservoir using function for opcode {res.name}")
            return inp_dim, out_dim, res
        else:
            raise ValueError(f"Opcode {opcode} not found in rnn_lib or self.funcs")
