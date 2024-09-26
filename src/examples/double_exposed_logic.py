from examples.imports import Reservoir, solve, inputs, plotters, sp

verbose = False
o1, o2, s1, s2 = sp.symbols('o1 o2 s1 s2')

# parameters and pitchfork base
xw = 0.025
xf = 0.1
cx = 3/13
ax = -cx / (3 * xw**2)
pitchfork_bifurcation = (ax * (o1 ** 3)) + (cx * o1)

# shifting logic for each gate
logic = {
    'and': -0.1 + (s1 + xf) * (s2 + xf) / (2 * xf),
    'nand': 0.1 + (s1 + xf) * (-s2 - xf) / (2 * xf),
    'or': 0.1 + (s1 - xf) * (-s2 + xf) / (2 * xf),
    'nor': -0.1 + (s1 - xf) * (s2 - xf) / (2 * xf),
    'xor': 0.0 + (-s1) * (s2) / xf,
    'xnor': 0.0 + (s1) * (s2) / xf
}

currentlyRunning = 'nor'
logic_eqs = [
    sp.Eq(o1, 0.1 * (pitchfork_bifurcation + logic[currentlyRunning])),
    sp.Eq(o2, o1)
]

logic_inputs = inputs.high_low_inputs(4000)
reservoir: Reservoir = Reservoir.solve(logic_eqs)

# display octave code (matlab readable equations)
if verbose:
    for eq in logic_eqs:
        print(sp.octave_code(eq))

# run network forward
outputs = reservoir.run4input(logic_inputs)
plotters.InOutSplit(logic_inputs, outputs, currentlyRunning + " Gate")

# save preset
if 1:
    reservoir.save(f"{currentlyRunning}_de")  
    res = Reservoir.load(f"{currentlyRunning}_de")
    res: Reservoir
