import ast as pyast


class ASTGenerator:
    def __init__(self):
        self.allowed_nodes = {
            # For imports
            pyast.ImportFrom,
            pyast.Attribute,
            pyast.alias,
            # Handled by pyres compiler
            pyast.Module,
            pyast.Expr,
            pyast.Assign,
            pyast.Name,
            pyast.Constant,
            pyast.FunctionDef,
            pyast.arguments,
            pyast.arg,
            pyast.Store,
            pyast.Tuple,
            pyast.Return,  # Return statement
            pyast.Call,  # Function call
            pyast.Load,
        }
        ast: pyast.Module = None

    def read_and_parse(self, filename) -> pyast.Module:
        with open(filename, "r") as file:
            source_code = file.read()
        self.ast = pyast.parse(source_code)
        self.validate()
        return self.ast

    def validate(self):
        validator = self.Validator(self.allowed_nodes)
        validator.visit(self.ast)

    def print(self):
        print(pyast.dump(self.ast, indent=4))

    class Validator(pyast.NodeVisitor):
        def __init__(self, allowed_nodes):
            self.allowed_nodes = allowed_nodes

        def generic_visit(self, node):
            if type(node) not in self.allowed_nodes:
                raise SyntaxError(f"Disallowed node type: {type(node).__name__}")
            super().generic_visit(node)
