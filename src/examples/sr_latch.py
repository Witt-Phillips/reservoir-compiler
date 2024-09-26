from examples.imports import Reservoir, solve, inputs, plotters
from prnn.circuit import Circuit

nor: Reservoir = Reservoir.load("nor")
nor_de = nor.copy().doubleOutput(0)

circuitRes = Circuit(
    [[nor, 0, nor_de, 0], [nor_de, 0, nor, 0]],
).connect()

inp = inputs.sr_inputs(2000)
outputs = circuitRes.run4input(inp)

plotters.InOutSplit(inp, outputs, "SR Latch")
