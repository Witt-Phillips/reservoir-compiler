import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from prnn.reservoir import *
from prnn.circuit import Circuit
from utils import plotters, inputs

nor: Reservoir = Reservoir.load("nor")
nor_de = nor.copy().doubleOutput(0)

circuitRes = Circuit(
    [[nor, 0, nor_de, 0],
    [nor_de, 0, nor, 0]],
    ).connect()

inp = inputs.sr_inputs(2000)
outputs = circuitRes.run4input(inp)

plotters.InOutSplit(inp, outputs, "SR Latch")


