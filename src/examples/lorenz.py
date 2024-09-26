from examples.imports import Reservoir, solve, inputs, plotters, sp


o1, o2, o3 = sp.symbols('o1 o2 o3')

lorenz_eqs = [
    sp.Eq(o1, o2 - o1),
    sp.Eq(o2, 1/10*o1 - 1/10*o2 - 20*o1*o3),
    sp.Eq(o3, 20*o1*o2 - 4/15*o3 - 0.036)
]

time = 5000
lorenz_inputs = inputs.zeros(time)

reservoir: Reservoir = Reservoir.solve(lorenz_eqs)

outputs = reservoir.run4input(lorenz_inputs)
plotters.threeD(outputs, "Lorenz Attractor")

# save preset
if 1:
    name = "lorenz"
    reservoir.save(name)  
    res = Reservoir.load(name)
    res: Reservoir



