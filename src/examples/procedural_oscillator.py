import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import *
from prnn_method import circuit
from utils import plotters

# Circuit configuration and generation.
nand: Reservoir = Reservoir.loadFile("nand")
nand1 = nand.copy()
nand2 = nand.copy()
nand3 = nand.copy()

# TODO: Make circuit wiring match Verilog conventions
# (outputNet, o#, inputNet, x#)

oscillator_circuit = [
    [nand1, 1, nand2, 1],
    [nand1, 1, nand2, 2],
    [nand2, 1, nand3, 1],
    [nand2, 1, nand3, 2],
    [nand3, 1, nand1, 1],
    [nand3, 1, nand1, 2]
]

circuitRes = circuit.connect(oscillator_circuit, [nand1, nand2, nand3])

#TODO: Remove list of reservoir argument to connect(). Read from circuit instead.


# Run for no input.
time = 7000
input_data = np.zeros((3, time, 4))
radp = circuitRes.run4input(input_data, np.identity(3*nand.A.shape[0])) # W = I to allow manual calculation below

# Find o=Wr
W = nand.W
outputs = np.vstack([
    W @ radp[0:30, :],  # first nand
    W @ radp[30:60, :],  # second nand
    W @ radp[60:90, :]   # third nand
])

# plot using 'outputs' to see all three signals
exposed_output = outputs[2, :].reshape(1, -1)
plotters.Outputs(exposed_output, "Procedurally Constructed Oscillator")