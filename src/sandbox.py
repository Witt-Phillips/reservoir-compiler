from reservoir import *
from prnn_method import circuit
from utils import plotters

# Circuit configuration and generation.
nand1 = Reservoir.gen_baseRNN(5, 2)
nand2 = Reservoir.gen_baseRNN(5, 2)
nand3 = Reservoir.gen_baseRNN(5, 2)

nand1.A += 1
nand1.B += 2
nand2.B = nand3.B = nand1.B
nand2.A = nand3.A = nand1.A

nand1.W = np.random.rand(1, 5)
nand2.W = np.random.rand(1, 5)
nand3.W = np.random.rand(1, 5)

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