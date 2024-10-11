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
        Expr(Opc.LET, [["l1a", "l2a", "l3a"], Expr("LORENZ", [])]),
        # rotate lorenz
        Expr(Opc.LET, [["l1", "l2", "l3"], Expr("LORENZ", [])]),
        Expr(Opc.LET, [["r1", "r2", "r3"], Expr("ROTATE90", ["l1", "l2", "l3"])]),
        Expr(Opc.RET, [["r1", "r2", "r3"]]),
        Expr(Opc.RET, [["l1a", "l2a", "l3a"]]),
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
        Expr(Opc.INPUT, [["i1", "i2", "i3"]]),
        Expr(Opc.LET, [["i1"], 1.0]),
        Expr(Opc.LET, [["a"]]),
        Expr(Opc.LET, [["b"], Expr("NAND", ["a", "i2"])]),
        Expr(Opc.LET, [["a"], Expr("NAND", ["i3", "i1"])]),
        Expr(Opc.RET, ["b"]),
    ]
)

simple_constant_inputs = Prog(
    [
        Expr(Opc.INPUT, [["i1", "i2"]]),
        Expr(Opc.LET, [["i1"], 1.0]),
        Expr(Opc.LET, [["i2"], 1.0]),
        Expr(Opc.LET, [["nando1"], Expr("NOR", ["i1", "i2"])]),
        Expr(Opc.RET, [["nando1"]]),
    ]
)

sr_latch = Prog(
    [
        Expr(Opc.INPUT, [["set", "reset"]]),
        Expr(Opc.LET, [["r2"]]),
        Expr(Opc.LET, [["Q", "r1"], Expr("NOR_DE", ["set", "r2"])]),
        Expr(Opc.LET, [["Qp", "r2"], Expr("NOR_DE", ["reset", "r1"])]),
        Expr(Opc.RET, [["Q", "Qp"]]),
    ]
)

graph: CGraph = Core(oscillator, verbose=False).compile_to_cgraph()
# graph.print()
graph.draw()
res = Resolver(graph, verbose=True).resolve()
# print("inputs: ", res.input_names)
# print("outputs: ", res.output_names)
# res.print()
inp = np.zeros((1, 4000))
# inp = inputs.high_low_inputs(4000)
outputs = res.run4input(inp)
plotters.plt_outputs(outputs, "osc", res.output_names)
# plotters.in_out_split(
#     inp, outputs, "sr_latch", input_names=res.input_names, output_names=res.output_names
# )

# save preset
# if 1:
#     res.name = "sr_latch"
#     res.save(res.name)

# plotters.plt_outputs(outputs, "sr_latch", res.output_names)
# plotters.plt_outputs(outputs, "lorenz", res.output_names)

# TIME = 4000
# lorenz_inputs = inputs.zeros(TIME)
# outputs = res.run4input(lorenz_inputs)
# plotters.plot_matrix_heatmap(res.A)
# plotters.plot_reservoir_matrices(res, "Lorenz Attractor")
# plotters.three_d(outputs, "Lorenz Attractor")
# plotters.three_d_input_output(outputs[:3, :], outputs[3:, :], "Lorenz Attractor")


# for node in graph.graph.nodes:
#     node = graph.get_node(node)
#     print(node)
#     if node["type"] == "reservoir":
#         print("inputs: ", node["reservoir"].input_names)
#         print("outputs: ", node["reservoir"].output_names)
# if node["type"] == "reservoir":
#     print("inputs: ", node["reservoir"].input_names)
