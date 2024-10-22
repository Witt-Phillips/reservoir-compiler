import ast
import re
from typing import List, Tuple
from _cgraph.cgraph import CGraph
from _prnn.reservoir import Reservoir
from _cgraph.resolve import Resolver
from _std.std import registry

from typing import Union
from dataclasses import dataclass


@dataclass
class CVec:
    """
    Compound vector: fundamental data type that wraps all primitives. List of floats, bools, or strs (syms)
    """

    cvec: list[Union[float, bool, str]]


class FnInfo:
    def __init__(self):
        self.res: Reservoir = None
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.vars: set = set()
        self.graph = CGraph()


class ASTCompiler(ast.NodeVisitor):
    def __init__(self, verbose=False):
        self.uid_ct = 0
        self.funcs: dict[str, FnInfo] = {}
        self.verbose: bool = verbose
        # Per function
        self.curr_fn: str = None

    def compile(self, prog_ast: ast.Module) -> Reservoir:
        """Takes source code and attempts to resolve it to a reservoir.
        Returns None on failure.
        """
        if self.verbose:
            print(ast.dump(prog_ast, indent=4))
        self.visit(prog_ast)

        for fn in self.funcs:
            self.funcs[fn].graph.validate()

        # print graphs
        if self.verbose:
            for fn in self.funcs:
                print(fn)
                graph = self.funcs[fn].graph
                graph.draw()
                graph.print()

        # resolve every function's cgraph to a reservoir
        made_resolution = True
        while made_resolution:
            made_resolution = False
            for fn in self.funcs:
                if self.funcs[fn].res is not None:
                    continue

                unresolved_res_ct = 0
                for node_name in self.funcs[fn].graph.all_nodes():
                    fn_name = self.strip_uid(node_name)
                    # get rid of uid tage
                    node = self.funcs[fn].graph.get_node(node_name)
                    if node["type"] != "reservoir":
                        continue

                    if node["reservoir"] is None:
                        # check in stdlib
                        if fn_name in registry:
                            node["reservoir"] = Reservoir.load(registry[fn_name])
                            made_resolution = True
                        # check in program defined fns
                        elif fn_name in self.funcs.keys():
                            res = self.funcs[fn_name].res
                            if res is not None:
                                node["reservoir"] = res
                                made_resolution = True
                        else:
                            unresolved_res_ct += 1

                # If all constituent reservoirs in the graph have been resovled, resolve the graph
                if unresolved_res_ct == 0:
                    self.funcs[fn].graph
                    self.funcs[fn].res = Resolver(self.funcs[fn].graph).resolve()
                    # TODO: make name tracking entirely graph based
                    self.funcs[fn].res.input_names = self.funcs[fn].inputs
                    self.funcs[fn].res.output_names = self.funcs[fn].outputs
                    made_resolution = True

        if "main" not in self.funcs:
            raise ValueError("Couldn't find 'main' function")
        res = self.funcs["main"].res

        # if self.verbose:
        #     res.print()

        return res

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # get name
        fn_name = node.name
        self.curr_fn = fn_name
        if self.verbose:
            print(f"Visiting function definition: {fn_name}")

        # create entry in funcs
        fn_info = FnInfo()
        self.funcs[fn_name] = fn_info

        # process args
        for arg in node.args.args:
            arg_name = arg.arg
            fn_info.inputs.append(arg_name)
            fn_info.graph.add_input(arg_name)
            if self.verbose:
                print(f"Declared input variable: {arg_name}")

        # iterate through body statements
        for stmt in node.body:
            self._process_statement(stmt)

    # TODO: what should return handle? Pyhton way: a tuple of outputs?
    # So we just enforce that you have to bind an expression to values before you return
    # them? Or do we allow returning arbitrary expressions?
    def visit_Return(self, node: ast.Return) -> None:
        """Return statements determine which variables become outputs
        ASSUMES: arguments to return are of name type.
        TODO: Add support for returning constants?
        """
        value: ast.AST = node.value
        values = [node.value] if isinstance(node.value, ast.Name) else node.value.elts
        for value in values:
            match value:
                case ast.Name():
                    name = value.id
                    self.funcs[self.curr_fn].graph.add_output(name)
                    self.funcs[self.curr_fn].outputs.append(name)
                    if self.verbose:
                        print(f"returning: {name}")
                case _:
                    ValueError("return: attempted to return unsupported type")

    def visit_Assign(self, node: ast.Assign) -> None:
        # get variable names as a list of variable names
        vars = []
        for target in node.targets:
            match target:
                case ast.Name(id=name):
                    vars.append(name)

                case ast.Tuple(elts=elts) | ast.List(elts=elts):
                    for elt in elts:
                        if isinstance(elt, ast.Name):
                            vars.append(elt.id)

        # process rhs expr and bind to vars
        cexpr = self._processs_expr(node.value)
        match cexpr:
            case None:
                for var in vars:
                    self.funcs[self.curr_fn].graph.add_var(var)

                    if self.verbose:
                        print(f"Forward declared {var}")

            case str():
                # Expr resolved to reservoir: bind outputs of called fn to vars
                res_name = cexpr
                for i, var in enumerate(vars):
                    # declare var
                    self.funcs[self.curr_fn].graph.add_var(var)

                    self.funcs[self.curr_fn].graph.add_edge(res_name, var, out_idx=i)
            case CVec():
                # Expr resolved to constant/ var list.
                consts = cexpr.cvec
                for i, var in enumerate(vars):
                    const = consts[i]
                    self.funcs[self.curr_fn].graph.add_var(var)

                    if var not in self.funcs[self.curr_fn].inputs:
                        self.funcs[self.curr_fn].inputs.append(var)
                        self.funcs[self.curr_fn].graph.add_input(var, val=const)
                    else:
                        node = self.funcs[self.curr_fn].graph.get_node(var)
                        node["value"] = const

                    if self.verbose:
                        print(f"Bound constant {const} to input var {var}")

    def visit_Call(self, node: ast.Call) -> str:
        func_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr

        res_name = self.uid_of_name(func_name)
        self.funcs[self.curr_fn].graph.add_reservoir(res_name, None)

        if self.verbose:
            print(f"Calling function {res_name}")

        for i, arg in enumerate(node.args):
            match arg:
                case ast.Name():
                    name = arg.id
                    # variable as argument
                    if (name not in self.funcs[self.curr_fn].inputs) and (
                        name not in self.funcs[self.curr_fn].vars
                    ):
                        ValueError(f"Used undefined symbol {name}")

                    self.funcs[self.curr_fn].graph.add_edge(name, res_name, in_idx=i)

                case ast.Constant():
                    name = self.uid_of_name(f"{arg.value}")
                    self.funcs[self.curr_fn].graph.add_input(name, val=arg.value)
                    self.funcs[self.curr_fn].graph.add_edge(name, res_name, in_idx=i)

        return res_name

    def _process_statement(self, node: ast.AST) -> None:  # what is the type?
        match node:
            case ast.FunctionDef():
                ValueError("Nested function declarations currently unsupported.")
            case ast.Return():
                self.visit_Return(node)
            case ast.Assign():
                self.visit_Assign(node)
            case _:
                ValueError("rest of process_statement unimplemented")
        pass

    def _processs_expr(self, node: ast.AST) -> Union[str, CVec, None]:
        match node:
            case ast.Constant():
                return self.visit_Constant(node)
            case ast.Call():
                return self.visit_Call(node)
            case ast.BinOp():
                ValueError("expr: binary operations not yet supported")
            case _:
                ValueError("res of expr unimplemented")
        pass

    def visit_Constant(self, node: ast.Constant) -> Union[CVec, None]:
        match node.value:
            case float():
                return CVec([node.value])
            case bool():
                val = 0.1 if node.value is True else -0.1
                return CVec([val])
            case None:
                return None
            case _:
                ValueError("constant: got invalid constant type")

    def uid_of_name(self, name: str) -> str:
        self.uid_ct += 1
        return f"{name}_{self.uid_ct}"

    def strip_uid(self, name_w_uid: str) -> str:
        return re.sub(r"_\d+$", "", name_w_uid)
