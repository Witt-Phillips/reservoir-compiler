from _prnn.reservoir import Reservoir
from _utils import inputs, plotters
import sympy as sp

""" NOTE: currently broken, working on figuring out why. """

o1, o2, o3 = sp.symbols("o1 o2 o3")

sprott_eqs = [
    sp.Eq(o1, -8 * o2),
    sp.Eq(o2, (25 / 4) * o1 + 5 * (o3**2)),
    sp.Eq(o3, 5 / 4 + 20 * o2 - 10 * o3),
]

if 0:
    for eq in sprott_eqs:
        print(sp.octave_code(eq))

time = 2000
sprott_inputs = inputs.zeros(time)

reservoir: Reservoir = Reservoir.solve(sprott_eqs)

outputs = reservoir.run(sprott_inputs)
plotters.three_d(outputs, "sprott Attractor")

# save preset
if 1:
    name = "sprott"
    reservoir.save(name)
    res = Reservoir.load(name)
    res: Reservoir
