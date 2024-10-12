from _prnn.reservoir import Reservoir
from _utils import inputs, plotters
from _prnn.circuit import Circuit

nor: Reservoir = Reservoir.load("nor")
nor_de = nor.copy().doubleOutput(0)

circuitRes = Circuit(
    [[nor, 0, nor_de, 0], [nor_de, 0, nor, 0]],
).connect()

inp = inputs.sr_inputs(2000)
outputs = circuitRes.run(inp)

plotters.in_out_split(inp, outputs, "SR Latch")
