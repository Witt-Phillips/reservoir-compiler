from frontend.res_ast import ASTGenerator
from frontend.ast_compiler import ASTCompiler
from prnn.reservoir import Reservoir

import numpy as np
from utils.plotters import plt_outputs


def main():
    ast = ASTGenerator().read_and_parse("src/frontend/example.pyres")
    res: Reservoir = ASTCompiler(verbose=True).compile(ast)

    inp = np.zeros((1, 4000))
    outputs = res.run4input(inp)
    plt_outputs(outputs, "Nand Series", res.output_names)


if __name__ == "__main__":
    main()
