import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from prnn.reservoir import *
from prnn.circuit import Circuit
from utils import plotters, inputs

# Circuit configuration and generation.
nand1: Reservoir = Reservoir.load("nand").doubleOutput(0) # creates exposed outputs on each nand
nand2 = nand1.copy()
#nand3 = Reservoir.load("nand_de")
nand3 = nand1.copy()

circuit = Circuit([
    [nand1, 0, nand2,0],
    [nand1, 0, nand2,1],
    [nand2, 0, nand3,0],
    [nand2, 0, nand3,1],
    [nand3, 0, nand1,0],
    [nand3, 0, nand1,1]],
    preserve_reservoirs=True)

circuitRes = circuit.connect()

# Run for no input.
time = 4000
input_data = np.zeros((1, time))
outputs = circuitRes.run4input(input_data)

plotters.Outputs(outputs, "Procedurally Constructed Oscillator")