""" Core class & compile() method """

from typing import Optional, Union, List, Tuple
import numpy as np
from ir.lang import Prog, Expr, Opc, Operand
from ir.fn_library import rnn_lib
from ir.variables import VariableManager
from prnn.reservoir import Reservoir
from prnn.circuit import Circuit


class Core:
    """
    Implements the core of the reservoir language compiler, compiling IR expressions -> Circuit
    """

    def __init__(self, prog: Prog):
        # circuit elements
        self.config: List[Tuple[Reservoir, int, Reservoir, int]] = []
        self.reservoirs: List[Reservoir] = {}
        self.readouts = []
        self.var_mngr = VariableManager()
        self.unresolved_exprs: List[Tuple[List[str], Expr]] = []  # deferred expressions
        self.resolved_deps = False  # flag to track new resolved expressions
        self.dependencies = set()  # tracks dependencies
        self.prog: Prog = prog

    def compile(self, verbose=True) -> Optional[Reservoir]:
        """
        Inputs: prog
        Outputs: None if circuit not well formed (will likely throw an error beforehand),
        connected reservoir otherwise.
        """
        for expr in self.prog.exprs:
            self._process_expr(expr)

        self._resolve_unresolved_expressions()

        circ = Circuit(self.config, [], reservoirs=list(self.reservoirs.keys()))

        if verbose:
            self._print_verbose_output()

        if not self.reservoirs:
            return None
        return circ.connect()

    def _process_expr(self, expr: Expr) -> Optional[Reservoir]:
        """
        * If an expression is a gate (operation on values), resolves to a
        reservoir by evaluating variable & inputs. Connects circuit accordingly.
        * In the case of presets (INPUT, LET, RET, etc.) returns None, indicating
        a top-level, non-gate operation.
        * If an expression relies on forward-declared variables, returns the
        maximally-evaluated expression, which will be pushed on to unresolved_exprs
        along with the expression's known dependencies.

        Input: expr (Expr): The expression containing the operation and operands.
        Output: Optional[Reservoir]: Reservoir object for custom opcodes, otherwise None.
        """
        match expr.op:
            case Opc.LET:
                return self._handle_let(expr)
            case Opc.REC:
                return self._handle_rec(expr)
            case Opc.INPUT:
                return self._handle_input(expr)
            case Opc.RET:
                return self._handle_ret(expr)
            case _:
                return self._handle_custom_opcode(expr)

    def _handle_let(self, expr: Expr):
        """
        * Add each variable name in operand 1 to the list of variables.
        * If defining a forward-declared variable, remove from forward-declaration list.
        """
        names, val = expr.operands
        processed_val = self._process_operand(val)

        if isinstance(processed_val, Reservoir):
            assert processed_val.W.shape[0] == len(names), "LET dimension mismatch"

        for i, name in enumerate(names):
            if self._is_dependency(name):
                self.var_mngr.forward_declare(name)
            else:
                self.var_mngr.declare_var(name, processed_val, i)
                self._resolve_deferred_expressions(name)

    def _handle_rec(self, expr: Expr):
        """
        * Forward define the variables in operand 1. Forward-defined variables
        are considered in scope for the user, but must be later defined to compile.
        """
        for name in expr.operands[0]:
            self.var_mngr.forward_declare(name)

    def _handle_input(self, expr: Expr):
        """
        * Define the input variables.
        """
        for name in expr.operands[0]:
            self.var_mngr.declare_input(name)

    def _handle_ret(self, expr: Expr):
        """
        * Return the readout variables (exposed values).
        * Ensure that the returned variables are not inputs.
        """
        for name in expr.operands[0]:
            resolved = self._process_operand(name)
            assert isinstance(resolved, tuple) and isinstance(
                resolved[0], Reservoir
            ), f"RET: invalid operand {name}"
            self.readouts.append(resolved)

    def _handle_custom_opcode(self, expr: Expr) -> Reservoir:
        """
        * Custom opcode handling:
        * Loads reservoir from opcode library and connects inputs accordingly.
        """
        opcode = expr.op
        assert isinstance(opcode, str), "Custom opcode must be a string"
        assert opcode in rnn_lib, f"Opcode {opcode} not found in library"

        _, _, res_path = rnn_lib[opcode]
        res = Reservoir.load(res_path).copy()
        self.reservoirs[res] = {"u_inp": 0, "u_out": 0}

        operands = expr.operands
        res_inp_dim = (
            0
            if not isinstance(res.x_init, np.ndarray) or res.x_init.ndim == 0
            else res.x_init.shape[0]
        )
        assert len(operands) == res_inp_dim, f"Opcode {opcode} input dimension mismatch"

        for i, operand in enumerate(operands):
            operand_resolved = self._process_operand(operand)
            self._connect_reservoirs(operand_resolved, res, i)

        return res

    def _process_operand(self, opr: Operand) -> Union[str, Reservoir]:
        """
        * Resolves an operand to either a string (input variable) or a reservoir
        (evaluated expression).
        * Handles direct variable lookups, input counting, and expression evaluation.

        Input:
        - opr (Operand): The operand to be processed, which can be a string (variable or input)
        or an expression.
        Output: Union[str, Reservoir]: The operand as a string if it's an input, or a Reservoir
        object if it represents a gate or expression.
        """
        if isinstance(opr, str):
            return self.var_mngr.get_var(opr)
        if isinstance(opr, Expr):
            return self._process_expr(opr)

        raise ValueError(f"Unsupported operand type: {type(opr)}")

    def _connect_reservoirs(self, operand, target_res, input_idx):
        """
        * Handles connecting reservoirs, ensuring that inputs and outputs
        are accounted for. This function links the output of one reservoir to the input
        of another.
        """
        if isinstance(operand, str):
            if operand in self.var_mngr.inps:
                self.reservoirs[target_res]["u_inp"] += 1
            elif operand in self.var_mngr.fwd_vars:
                self.dependencies.add(operand)
            else:
                raise ValueError(f"Undefined operand {operand}")
        elif isinstance(operand, tuple):
            source_res, output_num = operand
            self.config.append([source_res, output_num, target_res, input_idx])
            self.reservoirs[target_res]["u_inp"] += 1
            self.reservoirs[source_res]["u_out"] += 1

    def _resolve_deferred_expressions(self, name):
        """
        * Resolves deferred expressions once forward-defined variables
        become defined.
        """
        for deps, _ in self.unresolved_exprs:
            if name in deps:
                deps.remove(name)
                self.resolved_deps = True
        if name in self.var_mngr.fwd_vars:
            self.var_mngr.rm_fwd_dec(name)

    def _resolve_unresolved_expressions(self):
        """
        * Second pass to handle unresolved expressions.
        * Iterates through the list of deferred expressions to resolve them once
        dependencies are satisfied.
        """
        while self.unresolved_exprs:
            deps, uexpr = self.unresolved_exprs.pop(0)
            if not deps:
                self._process_expr(uexpr)
            else:
                self.unresolved_exprs.append((deps, uexpr))
            if self.resolved_deps:
                self.resolved_deps = False
                break

    def _is_dependency(self, name):
        """
        * Check if a given name is part of the current dependencies.
        """
        return self.dependencies and name in self.dependencies

    def _print_verbose_output(self):
        """
        * Prints verbose information about the circuit configuration, including
        connections, reservoirs, variables, inputs, readouts, forward variables,
        and unresolved expressions.
        """
        print("Config:\n" + "\n".join(str(connection) for connection in self.config))
        print(
            "\nReservoirs:\n"
            + "\n".join(str(reservoir) for reservoir in self.reservoirs)
        )
        print(
            "\nVars:\n" + "\n".join(f"{k}: {v}" for k, v in self.var_mngr.vars.items())
        )
        print(
            "\nInputs:\n"
            + "\n".join(f"{k}: {v}" for k, v in self.var_mngr.inps.items())
        )
        print("\nReadout:\n" + "\n".join(str(readout) for readout in self.readouts))
        print(
            "\nForward Vars:\n" + "\n".join(str(var) for var in self.var_mngr.fwd_vars)
        )
        print(
            "\nUnresolved Expressions:\n"
            + "\n".join(str(expr) for expr in self.unresolved_exprs)
        )
