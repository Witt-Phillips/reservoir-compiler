""" A full stack example of the PRNN method & IR compilation"""

import sympy as sp
from prnn.reservoir import Reservoir

""" 1. Define constituent reservoirs ------------------------------ """
# Lorenz attractor
o1, o2, o3 = sp.symbols("o1 o2 o3")

lorenz_eqs = [
    sp.Eq(o1, o2 - o1),
    sp.Eq(o2, 1 / 10 * o1 - 1 / 10 * o2 - 20 * o1 * o3),
    sp.Eq(o3, 20 * o1 * o2 - 4 / 15 * o3 - 0.036),
]

lorenz_res = Reservoir.solve(lorenz_eqs)
print("Solved Lorenz")

# pi/2 rotation
o1, o2, o3, s1, s2, s3 = sp.symbols("o1 o2 o3 s1 s2 s3")

rotation_eqs = [sp.Eq(o1, -s2), sp.Eq(o2, s1), sp.Eq(o3, s3)]

rotation_res = Reservoir.solve(rotation_eqs)
print("Solved Rotation")

""" 2. Define program & compile to Graph ------------------------------ """
from ir.core import Core, Expr, Opc
from ir.lang import Prog

lorenz_rotation = Prog(
    [
        # reference lorenz
        Expr(Opc.LET, [["l1a", "l2a", "l3a"], Expr("custom_lorenz", [])]),
        # rotate lorenz
        Expr(Opc.LET, [["l1", "l2", "l3"], Expr("custom_lorenz", [])]),
        Expr(
            Opc.LET, [["r1", "r2", "r3"], Expr("custom_rotation", ["l1", "l2", "l3"])]
        ),
        Expr(Opc.RET, [["r1", "r2", "r3"]]),
        Expr(Opc.RET, [["l1a", "l2a", "l3a"]]),
    ]
)


graph = Core(
    lorenz_rotation,
    funcs={
        "custom_lorenz": (0, 3, lorenz_res),
        "custom_rotation": (3, 3, rotation_res),
    },
).compile_to_cgraph()

# Visualize the CGraph
if False:
    graph.draw()

""" 3. Resolve graph into a reservoir ------------------------------ """
from cgraph.resolve import Resolver

res = Resolver(graph, verbose=True).resolve()

""" 4. Run reservoir & plot outputs ------------------------------ """
from utils.inputs import zeros
from utils.plotters import three_d_input_output

TIME = 4000
inp = zeros(TIME)
outputs = res.run4input(inp)
three_d_input_output(outputs[:3, :], outputs[3:, :], "RotatedLorenz Attractor")

""" Other visualization options """

from utils.plotters import plot_reservoir_matrices

# Visualization the reservoir matrices
if False:
    plot_reservoir_matrices(res, "Rotated Lorenz")
