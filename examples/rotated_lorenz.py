# from examples.imports import Reservoir, inputs, plotters, sp
from prnn.reservoir import Reservoir
from utils import inputs, plotters
import sympy as sp
from prnn.circuit import Circuit

rotation_res: Reservoir = Reservoir.load("rotation90")
lorenz_res: Reservoir = Reservoir.load("lorenz")

lorenz_res.print()

rotated_lorenz_res = Circuit(
    # output matrix, o#, rotation input, i#
    [
        [lorenz_res, 0, rotation_res, 0],
        [lorenz_res, 1, rotation_res, 1],
        [lorenz_res, 2, rotation_res, 2],
    ],
    preserve_reservoirs=True,
).connect()

lorenz_inputs = inputs.zeros(4000)
og_outputs = lorenz_res.run4input(lorenz_inputs)
rot_outputs = rotated_lorenz_res.run4input(lorenz_inputs)
plotters.plot_reservoir_matrices(rotated_lorenz_res, "Rotated Lorenz")
# plotters.three_d_input_output(og_outputs, rot_outputs, "Circuit: Lorenz -> Rotation")
