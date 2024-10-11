import ast
import re
from typing import List, Tuple
from cgraph.cgraph import CGraph
from prnn.reservoir import Reservoir
from cgraph.resolve import Resolver

from frontend.stdlib import registry


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
        print(ast.dump(prog_ast, indent=4))
        self.visit(prog_ast)

        # TODO: resolution loop: attempt to
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
                    self.funcs[fn].res.input_names = self.funcs[fn].inputs
                    self.funcs[fn].res.output_names = self.funcs[fn].outputs
                    made_resolution = True

        if "main" not in self.funcs:
            raise ValueError("Couldn't find 'main' function")

        return self.funcs["main"].res

    def visit_FunctionDef(self, node: ast.FunctionDef):
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

        self.funcs[fn_name].graph.print()

    def visit_Return(self, node: ast.Return):
        """Return statements determine which variables become outputs
        ASSUMES: arguments to return are of name type.
        TODO: Add support for returning constants?
        """
        value: ast.AST = node.value
        values = [node.value] if isinstance(node.value, ast.Name) else node.value.elts
        for value in values:
            if isinstance(value, ast.Name):
                name = value.id
                self.funcs[self.curr_fn].graph.add_output(name)
                if self.verbose:
                    print(f"returning: {name}")

    def visit_Assign(self, node: ast.Assign):
        vars = (
            [node.targets[0]]
            if isinstance(node.targets[0], ast.Name)
            else node.targets[0].elts
        )

        # get values
        match node.value:
            case ast.Constant():
                name = vars[0].id
                self.funcs[self.curr_fn].graph.add_var(name)

                const = self.visit_Constant(node.value)

                if name not in self.funcs[self.curr_fn].inputs:
                    self.funcs[self.curr_fn].inputs.append(name)
                    self.funcs[self.curr_fn].graph.add_input(name, val=const)
                else:
                    node = self.funcs[self.curr_fn].graph.get_node(name)
                    node["value"] = const

                if self.verbose:
                    print(f"Bound constant {const} to input var {name}")

            case ast.Call():
                res_name = self.visit_Call(node.value)

                # bind outputs of called fn to vars
                for i, var in enumerate(vars):
                    # declare var
                    var_name = var.id
                    self.funcs[self.curr_fn].graph.add_var(var_name)

                    self.funcs[self.curr_fn].graph.add_edge(
                        res_name, var_name, out_idx=i
                    )

        # bind output of function to vars
        # TODO: make this a more robust expr-eval mechanism. For now, we'll
        # just assume variables are always set to a function call or number

    def visit_Constant(self, node: ast.Constant) -> float:
        return node.value

    def visit_Call(self, node: ast.Call) -> str:
        res_name = self.uid_of_name(node.func.id)
        self.funcs[self.curr_fn].graph.add_reservoir(res_name, None)

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
                    name = self.uid_of_name("const_in")
                    self.funcs[self.curr_fn].graph.add_input(name, val=arg.value)
                    self.funcs[self.curr_fn].graph.add_edge(name, res_name, in_idx=i)

        return res_name

    def _process_statement(self, node: ast.AST):  # what is the type?
        match node:
            case ast.FunctionDef():
                ValueError("Nested function declarations currently unsupported.")
            case ast.Return():
                self.visit_Return(node)
            case ast.Assign():
                self.visit_Assign(node)
        pass

    def uid_of_name(self, name: str) -> str:
        self.uid_ct += 1
        return f"{name}_{self.uid_ct}"

    def strip_uid(self, name_w_uid: str) -> str:
        return re.sub(r"_\d+$", "", name_w_uid)
