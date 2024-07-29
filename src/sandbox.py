from reservoir import *
from prnn_method import circuit
from utils import plotters

# Circuit configuration and generation.
nand: Reservoir = Reservoir.loadFile("nand")

nand1 = nand.copy()
nand2 = nand.copy()
nand3 = nand.copy()

# TODO  Input -> Output
# to Output -> Input 

# sets the input number of net2 to the output number of net1
# (outputNet, o#, inputNet, x#)
oscillator_circuit = [
    [nand1, 1, nand2, 1],
    [nand1, 1, nand2, 2],
    [nand2, 1, nand3, 1],
    [nand2, 1, nand3, 2],
    [nand3, 1, nand1, 1],
    [nand3, 1, nand1, 2]
]

#TODO no need to pass in the gates again as they are already in the circuit
circuit.connect(oscillator_circuit, [nand1, nand2, nand3])

# inputs = np.zeros((1, 10, 4))
# outputs = oscillator_reservoir.run4input(inputs)
# plotters.InOutSplit(inputs, outputs, "Oscillator")