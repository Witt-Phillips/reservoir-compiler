import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import *
from prnn import circuit
from utils import plotters, inputs

nor_res = Reservoir.loadFile("nor")
nor1 = nor_res.copy()
nor2 = nor_res.copy()

sr_circuit = [
    [nor1, 1, nor2, 1],
    [nor2, 1, nor1, 1]
]

circuitRes = circuit.connect(sr_circuit, [nor1, nor2])
inp = inputs.sr_inputs(2000)
radp = circuitRes.run4input(inp, np.identity(2 * nor_res.A.shape[0]))

outputs = (nor_res.W @ radp[0:30, :]).reshape(1, -1)
plotters.InOutSplit(inp, outputs, "SR Latch")


