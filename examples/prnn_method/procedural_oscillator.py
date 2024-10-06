from prnn.reservoir import Reservoir
from utils import inputs, plotters
import sympy as sp
from prnn.circuit import Circuit

# Circuit configuration and generation.
nand1: Reservoir = Reservoir.load("nand").doubleOutput(
    0
)  # creates exposed outputs on each nand
nand2 = nand1.copy()
# nand3 = Reservoir.load("nand_de")
nand3 = nand1.copy()

circuit = Circuit(
    [
        [nand1, 0, nand2, 0],
        [nand1, 0, nand2, 1],
        [nand2, 0, nand3, 0],
        [nand2, 0, nand3, 1],
        [nand3, 0, nand1, 0],
        [nand3, 0, nand1, 1],
    ],
    preserve_reservoirs=True,
)

circuitRes = circuit.connect()
# Run for no input.
time = 4000
input_data = inputs.zeros(time)
outputs = circuitRes.run4input(input_data)

plotters.plt_outputs(outputs, "Procedurally Constructed Oscillator")
