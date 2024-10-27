from _frontend.res_ast import ASTGenerator
from _frontend.ast_compiler import ASTCompiler
from _prnn.reservoir import Reservoir


def compile(path: str, verbose=False) -> Reservoir:
    ast = ASTGenerator().read_and_parse(path)
    return ASTCompiler(verbose=verbose, file=path).compile(ast)
