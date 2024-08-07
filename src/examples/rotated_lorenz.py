import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from reservoir import Reservoir
from prnn.circuit import Circuit 
from utils import inputs, plotters

rotation_res: Reservoir = Reservoir.load("rotation90")
lorenz_res: Reservoir =   Reservoir.load("lorenz")

rotated_lorenz_res = Circuit(
    [[lorenz_res, 0, rotation_res, 0],
    [lorenz_res, 1, rotation_res, 1],
    [lorenz_res, 2, rotation_res, 2]],
    preserve_reservoirs=True).connect()

lorenz_inputs = inputs.zeros(4000)
og_outputs = lorenz_res.run4input(lorenz_inputs)
rot_outputs = rotated_lorenz_res.run4input(lorenz_inputs)

plotters.threeDInputOutput(og_outputs, rot_outputs, "Circuit: Lorenz -> Rotation")