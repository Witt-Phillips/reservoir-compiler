import ast
import re
import time
from typing import List, Tuple, NewType
from _cgraph.cgraph import CGraph
from _prnn.reservoir import Reservoir
from _cgraph.resolve import Resolver
from _std.std import registry

from typing import Union
from dataclasses import dataclass

cvec = NewType("cvec", List[Union[float, str, None]])


class FnInfo:
    def __init__(self):
        self.res: Reservoir = None
        self.out_dim: int = None
        self.inp_dim: int = None
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.vars: set = set()
        self.graph = CGraph()


class ASTCompiler(ast.NodeVisitor):
    def __init__(self, verbose=False, track_time=False):
        self.uid_ct = 0
        self.funcs: dict[str, FnInfo] = {}
        self.head: ast.Module = None
        self.verbose: bool = verbose
        self.track_time = track_time
        self.time_data = {}
        # Per function
        self.curr_fn: str = None

    def compile(self, prog_ast: ast.Module) -> Reservoir:
        """Takes source code and attempts to resolve it to a reservoir.
        Returns None on failure.
        """
        self.head = prog_ast

        if self.verbose:
            print(ast.dump(prog_ast, indent=4))

        if self.track_time:
            print()
            self._start_timer("traverse ast")

        # find entry point main() and compile
        self.visit(self._get_fn_node("main"))

        if self.track_time:
            self._end_timer("traverse ast")

        for fn in self.funcs:
            self.funcs[fn].graph.validate()

        # print graphs
        if self.verbose:
            for fn in self.funcs:
                print("VERBOSE INFO FOR FN: ", fn)
                graph = self.funcs[fn].graph
                graph.draw()
                graph.print()

        if self.track_time:
            self._start_timer("global resolution loop")

        # resolve every function's cgraph to a reservoir
        made_resolution = True
        while made_resolution:
            made_resolution = False
            for fn in self.funcs:
                if self.funcs[fn].res is not None:
                    continue

                if self.verbose:
                    print(f"ATTEMPTING TO RESOLVE {fn}")

                unresolved_res_ct = 0
                for node_name in self.funcs[fn].graph.all_nodes():
                    fn_name = self.strip_uid(node_name)
                    node = self.funcs[fn].graph.get_node(node_name)
                    if node["type"] != "reservoir":
                        continue

                    if node["reservoir"] is None:
                        # check in stdlib
                        print(f"found unresolved reservoir {fn_name}")
                        if fn_name in registry:
                            if self.verbose:
                                print(f"res loop: found {fn_name} in registry")

                            node["reservoir"] = Reservoir.load(registry[fn_name].path)
                            res = node["reservoir"]
                            res: Reservoir
                            # give placeholder names if none
                            if res.name == []:
                                res.name = fn_name
                            if res.output_names == []:
                                for i in range(res.W.shape[0]):
                                    res.output_names.append(f"{fn_name}_o{i}")
                            if res.input_names == []:
                                for i in range(res.x_init.shape[0]):
                                    res.input_names.append(f"{fn_name}_o{i}")

                            made_resolution = True
                        # check in program defined fns
                        elif fn_name in self.funcs.keys():
                            if self.verbose:
                                print(f"res loop: found {fn_name} in compiled fns")
                            res = self.funcs[fn_name].res
                            if res is not None:
                                node["reservoir"] = res.copy()  # to avoid overwrites
                                made_resolution = True
                            else:
                                unresolved_res_ct += 1
                        else:
                            unresolved_res_ct += 1

                # If all constituent reservoirs in the graph have been resovled, resolve the graph
                if unresolved_res_ct == 0:
                    self.funcs[fn].graph

                    if self.track_time:
                        self._start_timer(f"resolve {fn}")

                    self.funcs[fn].res = Resolver(self.funcs[fn].graph).resolve()

                    if self.verbose:
                        print(f"reservoir resolved: {fn}")
                        self.funcs[fn].res.print()

                    if self.track_time:
                        self._end_timer(f"resolve {fn}")

                    # TODO: make name tracking entirely graph based
                    if self.verbose:
                        print(f"Solved fn {fn} inputs: ", self.funcs[fn].inputs)
                        print(f"Solved fn {fn} outputs: ", self.funcs[fn].outputs)

                    self.funcs[fn].res.input_names = self.funcs[fn].inputs
                    self.funcs[fn].res.output_names = self.funcs[fn].outputs
                    made_resolution = True

        if self.track_time:
            self._end_timer("global resolution loop")

        if "main" not in self.funcs:
            raise ValueError("Couldn't find 'main' function")

        res = self.funcs["main"].res
        assert res is not None, "compile: failed to compile reservoir"
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

    # TODO: allow us to return an expression (cvec of variables only?)
    def visit_Return(self, node: ast.Return) -> None:
        """Return statements determine which variables become outputs
        ASSUMES: arguments to return are of name type.
        TODO: Add support for returning constants?
        """
        match node.value:
            case ast.Tuple():
                vars = cvec([])
                for elt in node.value.elts:
                    var_cvec = self._processs_expr(elt)
                    vars.extend(var_cvec)

                print("vars", vars)
            case _:
                vars = self._processs_expr(node.value)
                print("in else case")
                print("vars: ", vars)

        self.funcs[self.curr_fn].out_dim = len(vars)

        for var in vars:
            match var:
                case str():
                    print(f"adding var {var} to outputs")
                    self.funcs[self.curr_fn].graph.add_output(var)
                    self.funcs[self.curr_fn].outputs.append(var)
                    if self.verbose:
                        print(f"returning: {var}")
                case _:
                    ValueError(
                        f"Pyres only supports returning variables. Attempted to return f{var}"
                    )

        """   value: ast.AST = node.value
        values = [node.value] if isinstance(node.value, ast.Name) else node.value.elts
        self.funcs[self.curr_fn].out_dim = len(values)
        for value in values:
            match value:
                case ast.Name():
                    name = value.id
                    self.funcs[self.curr_fn].graph.add_output(name)
                    self.funcs[self.curr_fn].outputs.append(name)
                    if self.verbose:
                        print(f"returning: {name}")
                case _:
                    ValueError("return: attempted to return unsupported type") """

    # TODO: match cvec against none
    # TODO: check
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
        rhs = self._processs_expr(node.value)
        for el, var in zip(rhs, vars):
            match el:
                case None:
                    # forward declared var
                    self.funcs[self.curr_fn].graph.add_var(var)
                    if self.verbose:
                        print(f"Forward declared {var}")

                case str():
                    # overwrite old var with new name
                    # if var not already forward declared, overwrite old var.
                    self.funcs[self.curr_fn].graph.update_var_name(el, var)

                case float():
                    # set var to float; treat as input
                    self.funcs[self.curr_fn].graph.add_var(var)
                    if var not in self.funcs[self.curr_fn].inputs:
                        self.funcs[self.curr_fn].inputs.append(var)
                        self.funcs[self.curr_fn].graph.add_input(var, val=el)
                    else:
                        node = self.funcs[self.curr_fn].graph.get_node(var)
                        node["value"] = el

                    if self.verbose:
                        print(f"Bound constant {el} to input var {var}")

    # TODO : test
    def visit_Call(self, node: ast.Call) -> cvec:
        """
        Either retrieves or compiles called fn, creates new fn node in caller graph, processes inputs and returns dummy outputs
        """
        # 1. Either get dim from compiled fn, retrieve from stdlib, or compile
        match node.func:
            case ast.Name():
                # check already compiled
                func_name = node.func.id

                if func_name not in self.funcs:
                    if self.verbose:
                        print(f"Fn {func_name} called but not compiled; now compiling")
                    temp = self.curr_fn
                    self.visit_FunctionDef(self._get_fn_node(func_name))
                    self.curr_fn = temp

                if self.verbose:
                    print(f"Fn {func_name} compiled; retrieving from context")
                out_dim = self.funcs[func_name].out_dim
                inp_dim = self.funcs[func_name].inp_dim

            case _:
                func_name = node.func.attr

                if func_name in registry:
                    if self.verbose:
                        print(f"Fn {func_name} in stdlib; retrieving from registry")

                    out_dim = registry[func_name].out_dim
                    inp_dim = registry[func_name].inp_dim
                else:
                    ValueError(
                        f"Function {func_name} imported from stdlib but not found in registry"
                    )

        # Add reservoir node to graph
        if self.verbose:
            print(f"Calling function {func_name}")

        res_name = self.uid_of_name(func_name)
        self.funcs[self.curr_fn].graph.add_reservoir(res_name, None)

        # process arguments (must be Cvecs of dim 1)
        for i, arg in enumerate(node.args):
            evald_arg = self._processs_expr(arg)
            print("arg: ", arg)
            print("evald arg: ", evald_arg)

            assert (
                len(evald_arg) == 1
            ), f"argument {arg} passed to function {self.curr_fn} has invalid dimensions: {len(evald_arg)} != 1"
            val = evald_arg[0]

            match val:
                case str():
                    name = val
                    # variable as argument
                    if (name not in self.funcs[self.curr_fn].inputs) and (
                        name not in self.funcs[self.curr_fn].vars
                    ):
                        ValueError(f"Used undefined symbol {name}")

                    self.funcs[self.curr_fn].graph.add_edge(name, res_name, in_idx=i)

                case float():
                    name = self.uid_of_name(f"{val}")
                    self.funcs[self.curr_fn].graph.add_input(name, val=val)
                    self.funcs[self.curr_fn].graph.add_edge(name, res_name, in_idx=i)

        # generate placeholder outputs
        outputs: cvec = []
        for i in range(out_dim):
            out_name = self.uid_of_name(f"{func_name}_o")
            outputs.append(out_name)
            self.funcs[self.curr_fn].graph.add_var(out_name)
            self.funcs[self.curr_fn].graph.add_edge(res_name, out_name, out_idx=i)

        return outputs

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

    def _processs_expr(self, node: ast.AST) -> cvec:
        match node:
            case ast.Constant():
                return self.visit_Constant(node)
            case ast.Call():
                return self.visit_Call(node)
            case ast.Name():
                return cvec([node.id])
            case ast.BinOp():
                ValueError("expr: binary operations not yet supported")

        ValueError("rest of expr unimplemented")

    def visit_Constant(self, node: ast.Constant) -> cvec:
        match node.value:
            case float():
                return cvec([node.value])
            case bool():
                val = 0.1 if node.value is True else -0.1
                return cvec([val])
            case None:
                return cvec([None])
            case _:
                ValueError("constant: got invalid constant type")

    def uid_of_name(self, name: str) -> str:
        self.uid_ct += 1
        return f"{name}_{self.uid_ct}"

    def strip_uid(self, name_w_uid: str) -> str:
        return re.sub(r"_\d+$", "", name_w_uid)

    def _start_timer(self, key: str):
        """Start timer for a given key."""
        if self.track_time:
            self.time_data[key] = time.time()

    def _end_timer(self, key: str):
        """End timer for a given key and log duration."""
        if self.track_time and key in self.time_data:
            elapsed_time = time.time() - self.time_data[key]
            print(f"{key} took {elapsed_time:.4f} seconds")
            self.time_data[key] = elapsed_time

    def _get_fn_node(self, name: str) -> ast.FunctionDef:
        """Finds the function definition in the ast, returns FunctionDef node"""
        for node in self.head.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
            else:
                ValueError(f"Called undefined function {name}")
