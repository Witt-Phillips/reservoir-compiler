""" 
Sandbox environment for the development of the ReservoirComiler IR
"""

import numpy as np
from ir.lang import Prog, Expr, Opc
from ir.core import Core
from cgraph.cgraph import CGraph
from cgraph.resolve import Resolver
from utils import plotters, inputs

lorenz = Prog(
    [
        # reference lorenz
        # Expr(Opc.LET, [["l1a", "l2a", "l3a"], Expr("LORENZ", [])]),
        # rotate lorenz
        Expr(Opc.LET, [["l1", "l2", "l3"], Expr("LORENZ", [])]),
        Expr(Opc.LET, [["r1", "r2", "r3"], Expr("ROTATE90", ["l1", "l2", "l3"])]),
        Expr(Opc.RET, [["r1", "r2", "r3"]]),
        # Expr(Opc.RET, [["l1a", "l2a", "l3a"]]),
    ]
)

oscillator = Prog(
    [
        Expr(
            Opc.LET,
            [["nand1o1", "nand1o2", "nand2o1", "nand2o2", "nand3o1", "nand3o2"]],
        ),
        Expr(
            Opc.LET,
            [["nand1o1", "nand1o2", "o1"], Expr("NAND_TRIPLE", ["nand3o1", "nand3o2"])],
        ),
        Expr(
            Opc.LET,
            [["nand2o1", "nand2o2", "o2"], Expr("NAND_TRIPLE", ["nand1o1", "nand1o2"])],
        ),
        Expr(
            Opc.LET,
            [["nand3o1", "nand3o2", "o3"], Expr("NAND_TRIPLE", ["nand2o1", "nand2o2"])],
        ),
        Expr(Opc.RET, [["o1", "o2", "o3"]]),
    ]
)

# Tests our ability to define a variable, h
forward_dec = Prog(
    [
        Expr(Opc.INPUT, [["i1", "i2"]]),
        Expr(Opc.LET, [["a"]]),
        Expr(Opc.LET, [["b"], Expr("NAND", ["a", "i2"])]),
        Expr(Opc.LET, [["a"], Expr("NAND", ["b", "i1"])]),
    ]
)

graph: CGraph = Core(oscillator, verbose=False).compile_to_cgraph()
# graph.print()
graph.draw()
res = Resolver(graph, verbose=True).resolve()
# res.print()
inp = inputs.zeros(4000)
outputs = res.run4input(inp)
plotters.plt_outputs(outputs, "Procedural Oscillator")

""" TIME = 5000
lorenz_inputs = inputs.zeros(TIME)
outputs = res.run4input(lorenz_inputs)
# plotters.plot_matrix_heatmap(res.A)
# plotters.plot_reservoir_matrices(res, "Lorenz Attractor")
plotters.three_d(outputs, "Lorenz Attractor")
 """
