from lang import Prog, Expr, Opc
from circuitify import prog2circuit
from utils import plotters, inputs
import numpy as np
#input: for all operands in input, put into the input array.

#TODO: make it such that each variable maps to a unique output.

# Define the program
prog = Prog(exprs=[
    Expr(Opc.INPUT, ["w"]),
    Expr(Opc.INPUT, ["x"]),
    Expr(Opc.INPUT, ["y"]),
    Expr(Opc.LET, [["z"], Expr("NOR", ["x", "y"])]),
    Expr(Opc.LET, [["a"], Expr("NOR", ["z", "w"])])
    # Expr(Opc.RET, ["a"])
])

prog1 = Prog([
    Expr(Opc.INPUT, [["a", "b"]]),
    Expr(Opc.LET, [["c"], Expr("NAND", ["a", "b"])]),
    Expr(Opc.RET, ([["c"]]))
    ])

lorenz = Prog([
    # reference lorenz
    Expr(Opc.LET, [["l1a", "l2a", "l3a"], Expr("LORENZ", [])]),
    # rotate lorenz
    Expr(Opc.LET, [["l1", "l2", "l3"], Expr("LORENZ", [])]),
    Expr(Opc.LET, [["r1", "r2", "r3"], Expr("ROTATE90", ["l1", "l2", "l3"])])
    ])

# this case requires recurrent variables. ie, we can't define them before we use them

#plan for recursion: keep a stack of recursive variables, 
oscillator = Prog([
    Expr(Opc.INPUT, [["i1", "i2", "i3"]]),
    Expr(Opc.REC, [["r1", "r2"]]),
    Expr(Opc.LET, [["a1"], Expr("NAND", ["r1", "i1"])]),
    Expr(Opc.LET, [["r1"], Expr("NAND", ["i2", "i3"])])


])


res = prog2circuit(lorenz)
if prog2circuit is None:
    ValueError("could not construct valid reservoir from code")


# time = 4000
# input_data = np.zeros((1, 4000))
# outputs = res.run4input(input_data)
# og_lorenz = outputs[:3]
# rotated_lorenz = outputs[3:]

# plotters.threeDInputOutput(og_lorenz, rotated_lorenz, "Programmed Lorenz Rotation")