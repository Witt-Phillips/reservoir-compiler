""" 
Sandbox environment for the development of the ReservoirComiler IR
"""

import numpy as np
from ir.lang import Prog, Expr, Opc
from ir.core import Core
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
        # Forward define the outputs for nand1, nand2, nand3
        Expr(
            Opc.REC,
            [["nand1o1", "nand1o2", "nand2o1", "nand2o2", "nand3o1", "nand3o2"]],
        ),
        # Define the NAND gate for nand1 (two outputs)
        Expr(
            Opc.LET, [["nand1o1", "nand1o2"], Expr("NAND", ["nand3o1", "nand3o2"])]
        ),  # Inputs are nand3's outputs
        # Define the NAND gate for nand2 (two outputs)
        Expr(
            Opc.LET, [["nand2o1", "nand2o2"], Expr("NAND", ["nand1o1", "nand1o2"])]
        ),  # Inputs are nand1's outputs
        # Define the NAND gate for nand3 (two outputs)
        Expr(
            Opc.LET, [["nand3o1", "nand3o2"], Expr("NAND", ["nand2o1", "nand2o2"])]
        ),  # Inputs are nand2's outputs
    ]
)

forward_dec = Prog(
    [
        Expr(Opc.INPUT, [["i1", "i2", "i3"]]),
        Expr(Opc.REC, [["a"]]),
        Expr(Opc.LET, [["b"], Expr("NAND", ["i1", "i2"])]),
        Expr(Opc.LET, [["a"], Expr("NAND", ["b", "i3"])]),
    ]
)

res = Core(lorenz).compile(verbose=True)

assert res is not None, "could not construct valid reservoir from code"

TIME = 4000
input_data = np.zeros((1, 4000))
outputs = res.run4input(input_data)
og_lorenz = outputs[:3]
rotated_lorenz = outputs[3:]

plotters.three_d_input_output(og_lorenz, rotated_lorenz, "Programmed Lorenz Rotation")
