""" 
Sandbox environment for the development of the ReservoirComiler IR
"""

import numpy as np
from ir.lang import Prog, Expr, Opc
from ir.circuitify import prog2circuit
from utils import plotters

lorenz = Prog(
    [
        # reference lorenz
        Expr(Opc.LET, [["l1a", "l2a", "l3a"], Expr("LORENZ", [])]),
        # rotate lorenz
        Expr(Opc.LET, [["l1", "l2", "l3"], Expr("LORENZ", [])]),
        Expr(Opc.LET, [["r1", "r2", "r3"], Expr("ROTATE90", ["l1", "l2", "l3"])]),
    ]
)

oscillator = Prog(
    [
        Expr(Opc.INPUT, [["i1", "i2", "i3"]]),
        Expr(Opc.REC, [["r1", "r2"]]),
        Expr(Opc.LET, [["a1"], Expr("NAND", ["r1", "i1"])]),
        Expr(Opc.LET, [["r1"], Expr("NAND", ["i2", "i3"])]),
    ]
)

res = prog2circuit(lorenz)

assert res is not None, "could not construct valid reservoir from code"

TIME = 4000
input_data = np.zeros((1, 4000))
outputs = res.run4input(input_data)
og_lorenz = outputs[:3]
rotated_lorenz = outputs[3:]

plotters.threeDInputOutput(og_lorenz, rotated_lorenz, "Programmed Lorenz Rotation")
