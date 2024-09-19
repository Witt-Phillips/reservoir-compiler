from lang import *

def run(prog: Prog) -> List[Tuple[str, bool]]:
    symbol_table: Dict[str, bool] = {}
    used_vars: Dict[str, bool] = {}
    floating_expressions = []
    ret_vars: List[Union[str, bool]] = []
    
    # Evaluate an expression recursively
    def evaluate_expr(expr: Expr) -> bool:
        if expr.opcode in op_funcs:
            func = op_funcs[expr.opcode]
            evaluated_operands = []
            for operand in expr.operands:
                if isinstance(operand, bool):
                    evaluated_operands.append(operand)
                elif isinstance(operand, str):
                    if operand in symbol_table:
                        evaluated_operands.append(symbol_table[operand])
                    else:
                        raise ValueError(f"Variable '{operand}' used before assignment.")
                elif isinstance(operand, Expr):
                    # Recursively evaluate the nested expression
                    evaluated_operands.append(evaluate_expr(operand))
                else:
                    raise ValueError(f"Invalid operand type: {type(operand)}.")
            # Perform the operation
            return func(*evaluated_operands)
        else:
            raise ValueError(f"Unsupported operation: {expr.opcode}")
    
    # First pass: Assign variables and track usage
    for idx, expr in enumerate(prog.exprs):
        if expr.opcode == Op.LET:
            if len(expr.operands) != 2:
                raise ValueError(f"LET operation requires exactly two operands, got {len(expr.operands)} at expression {idx + 1}.")
            var_name, operation = expr.operands
            if not isinstance(var_name, str):
                raise ValueError(f"LET operation's first operand must be a variable name (str), got {type(var_name)} at expression {idx + 1}.")
            if not isinstance(operation, Expr):
                raise ValueError(f"LET operation must have an Expr as its second operand at expression {idx + 1}.")
            # Mark variables used in operands
            for operand in operation.operands:
                if isinstance(operand, str):
                    used_vars[operand] = True
                elif isinstance(operand, Expr):
                    # Optionally, traverse nested expressions to mark all used variables
                    def mark_used_vars(nested_expr: Expr):
                        for op in nested_expr.operands:
                            if isinstance(op, str):
                                used_vars[op] = True
                            elif isinstance(op, Expr):
                                mark_used_vars(op)
                    mark_used_vars(operand)
        elif expr.opcode == Op.RET:
            ret_vars = expr.operands
            # Mark RET variables as used
            for var in ret_vars:
                if isinstance(var, str):
                    used_vars[var] = True
        else:
            # It's an expression not assigned to any variable
            floating_expressions.append((idx + 1, expr))
    
    # Warn about floating expressions
    if floating_expressions:
        print("Warning: There are floating expressions that are not used or returned.")
        for expr_idx, expr in floating_expressions:
            print(f"  Floating expression at position {expr_idx}: {expr}")
    
    # Second pass: Evaluate expressions
    for idx, expr in enumerate(prog.exprs):
        if expr.opcode == Op.LET:
            var_name, operation = expr.operands
            # Evaluate the expression recursively
            result = evaluate_expr(operation)
            symbol_table[var_name] = result
        elif expr.opcode == Op.RET:
            # RET is handled after evaluation
            pass
        else:
            # Floating expressions are ignored or can raise warnings
            pass
    
    # Prepare the return values as list of tuples (var_name, value)
    results: List[Tuple[str, bool]] = []
    for var in ret_vars:
        if isinstance(var, str):
            if var in symbol_table:
                results.append((var, symbol_table[var]))
            else:
                raise ValueError(f"RET variable '{var}' is not defined.")
        elif isinstance(var, bool):
            # Optionally, handle literals differently
            results.append((str(var), var))  # Using str(var) as the name for literals
        else:
            raise ValueError("RET operands must be variables or boolean literals.")
    
    return results