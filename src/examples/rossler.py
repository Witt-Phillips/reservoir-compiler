from examples.imports import Reservoir, solve, inputs, plotters, sp

o1, o2, o3 = sp.symbols('o1 o2 o3')

rossler_eqs = [
    sp.Eq(o1, (-5*o2) - (5 * o3) - 3),
    sp.Eq(o2, (5 * o1) + o2),
    sp.Eq(o3, ((250/3) * o1 * o3) + (50*o1) - (28.5*o3) - 17.04)
]

time = 5000
rossler_inputs = inputs.zeros(time)

reservoir = Reservoir.solve(rossler_eqs)
reservoir: Reservoir

outputs = reservoir.run4input(rossler_inputs)
plotters.threeD(outputs, "Rossler Attractor")

# save preset
if 0:
    name = "rossler"
    reservoir.save(name)  
    res = Reservoir.load(name)
    res: Reservoir



