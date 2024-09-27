""" 
Converts a Program (list of Exprs) to its equivalent form as a Circuit object
"""

from typing import Optional, Union
import numpy as np
from ir.lang import Prog, Expr, Opc, Operand
from ir.fn_library import rnn_lib
from prnn.reservoir import Reservoir
from prnn.circuit import Circuit


def prog2circuit(prog: Prog, verbose=True) -> Optional[Reservoir]:
    """
    Inputs: prog
    Outputs: None if circuit not well formed (will likely throw an error beforehand),
    connected reservoir otherwise.
    """
    # circuit elements
    config = []
    reservoirs = {}
    readouts = []

    # compiler state
    variables = {}
    inputs = {}  # name -> # uses
    forward_variables = set()  # forward declared variables; unresolved
    unresolved_exprs = []  # stack unresolved expressions
    resolved_deps = False  # new resolved expr flag

    def process_expr(expr: Expr) -> Optional[Reservoir]:
        """
        * If an expression is a gate (operation on values), resolves to a
        reservoir by evaluating variable & inputs. Connects circuit accordingly.
        * In the case of presets (INPUT, LET, RET, etc.) returns None, indicating
        a top-level, non-gate operation.
        * If an expression relies on forward-declared variables, returns the
        maximally-evaluated expression, which will be pushed on to unresolved_exprs

        Input: expr (Expr): The expression containing the operation and operands.
        Output: Optional[Reservoir]: Reservoir object for custom opcodes, otherwise None.
        """
        opcode = expr.op
        operands = expr.operands

        match opcode:
            case Opc.LET:
                # Add each variable name in operand 1 to the list of variables.
                # If defining a forward-declared variable, remove from
                # forward-declaration list.
                # TODO: look into variable dependency graph for smoother
                # forward-dependence resolution.
                #####################################################################

                assert (
                    len(operands) == 2
                ), "LET: takes two args [name: str, value: operand]"
                names = operands[0]
                val = operands[1]
                processed_val = process_operand(
                    val
                )  # note: we choose to leave val unexpanded here; use lazy evalution instead

                if isinstance(processed_val, Reservoir):
                    processed_val: Reservoir
                    assert processed_val.W.shape[0] == len(
                        names
                    ), f"LET: {val.op} returns {processed_val.W.shape[0]} args, \
                        can't cast to {len(names)} variables"

                for i, name in enumerate(names):
                    assert isinstance(
                        name, str
                    ), f"LET: var name must be of type str. {name} is of type {type(name)}"
                    if dependencies:
                        forward_variables.add(name)
                    else:
                        variables[name] = (
                            processed_val,
                            i,
                        )  # store reservoir and output # to hash table.
                        # check to see if var was forward defined
                        # (& if we can resolve anything as a result)
                        for unresolved_pair in unresolved_exprs:
                            uexpr_deps, _ = unresolved_pair
                            if name in uexpr_deps:
                                resolved_deps = True

                        if name in forward_variables:
                            forward_variables.remove(name)
                return None
            case Opc.REC:
                # Forward define the variables in operand 1. Forward defined variables
                # are considered in scope for the user, but must be later defined to compile.
                # TODO: implement a method to find circular definition and stop
                # infinite-loop resolution
                #####################################################################
                assert len(operands) == 1, "REC: takes 1 arg [list[name: str]]"
                names = operands[0]
                for name in names:
                    assert isinstance(
                        name, str
                    ), f"REC: var name must be of type str. {name} is of type {type(name)}"
                    forward_variables.add(name)
                return None
            case Opc.INPUT:
                assert len(operands) == 1, "INPUT: takes one arg [list[name: str]]"
                names = operands[0]
                for name in names:
                    assert isinstance(
                        name, str
                    ), f"INPUT: input name must be of type str. {name} is of type {type(name)}"
                    assert (
                        name not in variables
                    ), f"INPUT: var already exists of name: {name}"
                    inputs[name] = 0
                return None
            case Opc.RET:
                # for now, we will leave ret empty and just return all exposed values.
                # later, ret will use "readouts" to tweak which outputs are suppressed/ exposed
                assert len(operands) == 1, "READOUT: takes one arg [list[name: str]]"
                names = operands[0]
                for name in names:
                    assert name not in inputs, "RET: cannot return a user input"
                    resolved = process_operand(name)
                    assert (
                        isinstance(resolved, tuple)
                        and isinstance(resolved[0], Reservoir)
                        and isinstance(resolved[1], int)
                    ), f"RET: could not resolve operand {name} to (Reservoir, int) \
                        pair, got {resolved} of type {type(resolved)}"
                    readouts.append(resolved)
                return None
            case _:
                # Custom opcode. Load from library
                # lookup reservoir from opcode
                # TODO: possible we can do opcode lookup and
                # load after analyzing for dependencies for efficiency?
                assert isinstance(
                    opcode, str
                ), f"CUSTOM_OPCODE: opcode name must be of type str. Got {type(opcode)}"
                assert (
                    opcode in rnn_lib
                ), f"CUSTOM_OPCODE: {opcode}: opcode not found in function library"
                _, _, res_path = rnn_lib[opcode]
                res: Reservoir = Reservoir.load(res_path).copy()

                # init each reservoir to have 0 inputs and 0 outputs used.
                reservoirs[res] = {"u_inp": 0, "u_out": 0}

                # eval inputs
                res_inp_dim = (
                    0
                    if not isinstance(res.x_init, np.ndarray) or res.x_init.ndim == 0
                    else res.x_init.shape[0]
                )
                assert (
                    len(operands) == res_inp_dim
                ), f"CUSTOM_OPCODE: {opcode}: passed {len(operands)} inputs to reservoir \
                      of input dim {res_inp_dim}"

                # Put operand values at inputs of target res.
                for i, operand in enumerate(operands):
                    operand = process_operand(operand)
                    match operand:
                        case str():
                            operand: str
                            if operand in inputs:
                                reservoirs[res]["u_inp"] += 1
                            elif operand in forward_variables:
                                dependencies.add(operand)
                                # reservoirs[res]['u_inp'] += 1
                            else:
                                raise ValueError(
                                    f"LET: {opcode} passed undefined operand {operand}"
                                )
                        case (Reservoir() as operand_res, int() as output_num):
                            config.append([operand_res, output_num, res, i])
                            reservoirs[res]["u_inp"] += 1
                            reservoirs[operand_res]["u_out"] += 1
                        case _:
                            raise ValueError(
                                f"CUSTOM_OPCODE: {opcode}: operand {i}:{operand} could not be \
                                simplified to reservoir/ input form"
                            )
                # if the expression cannot be resolved, abort for deferred resolution
                if dependencies:
                    return None
                return res

    def process_operand(opr: Operand) -> Union[str, Reservoir]:
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
        match opr:
            case str():
                if opr in variables:
                    return variables[opr]  # -> (reservoir, output#)
                return opr
                # CHECK: we do all of this processing in process_expr()?
                # elif opr in inputs:
                #     inputs[opr] += 1
                #     return opr # -> str
                # elif opr in forward_variables:
                #     print(f"process_operand found forward var {opr}")
                #     dependencies.add(opr)
                #     return opr
                # else:
                #     SyntaxError(f"{opr}: unbound var/ input.")
            case Expr():
                return process_expr(opr)  # -> None if deferred, Reservoir if resolved
            case _:
                raise ValueError(f"Unsupported operand type: {type(opr)}")

    # main processing loop
    for expr in prog.exprs:
        # resolve any dependencies possible
        if resolved_deps:
            resolved_deps = False
            dependencies = set()
            for upair in unresolved_exprs:
                udeps, uexpr = upair
                if not udeps:
                    result = process_expr(uexpr)
            if dependencies:
                unresolved_exprs.append((dependencies, uexpr))

        # processes the next expression
        dependencies = set()
        result = process_expr(expr)
        if dependencies:
            unresolved_exprs.append((dependencies, expr))

        assert (
            result is None
        ), f"Expression {expr} is floating. Be sure to bind it to a value."

    circ = Circuit(config, [], reservoirs=list(reservoirs.keys()))

    if verbose:
        print("Config:\n" + "\n".join(str(connection) for connection in config))
        print("\nReservoirs:\n" + "\n".join(str(reservoir) for reservoir in reservoirs))
        print("\nVars:\n" + "\n".join(f"{k}: {v}" for k, v in variables.items()))
        print("\nInputs:\n" + "\n".join(f"{k}: {v}" for k, v in inputs.items()))
        print("\nReadout:\n" + "\n".join(str(readout) for readout in readouts))
        print("\nForward Vars:\n" + "\n".join(str(var) for var in forward_variables))
        print(
            "\nUnresolved Expressions:\n"
            + "\n".join(str(expr) for expr in unresolved_exprs)
        )

    if not reservoirs:
        return None

    return circ.connect()
