import sympy as sp
from reservoir import *
from prnn_method import circuit, prnn_method
from utils import inputs, plotters

def main():
    verbose = False
    # define symbolic equations (naming is not constrained) -----------------
    o1, s1, s2 = sp.symbols('o1 s1 s2')
    
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

    currentlyRunning = 'nand'
    
    # list of symbolic output equations
    logic_eqs = [
        sp.Eq(o1, pitchfork_bifurcation + logic[currentlyRunning])
    ]
    
    # display octave code (matlab readable equations)
    if verbose:
        for eq in logic_eqs:
            print(sp.octave_code(eq))

    # generate inputs
    logic_inputs = inputs.high_low_inputs(1000)

    # solve for W/ internalize recurrencies (solveReservoir can also take & run inputs)
    nand_res, _ = Reservoir.solveReservoir(logic_eqs)
    nand_res: Reservoir

    # run network forward
    #outputs = nand_res.run4input(logic_inputs)
    #plotters.InOutSplit(logic_inputs, outputs, currentlyRunning + " Gate")

    """     # circuit experimentation
    nand1 = nand_res.copy()
    nand2 = nand_res.copy()
    nand3 = nand_res.copy()

    oscillator_circuit = [
    [nand1, 1, nand2, 1],
    [nand1, 1, nand2, 2],
    [nand2, 1, nand3, 1],
    [nand2, 1, nand3, 2],
    [nand3, 1, nand1, 1],
    [nand3, 1, nand1, 2]
]

    #TODO no need to pass in the gates again as they are already in the circuit
    oscillatorRes = circuit.connect(oscillator_circuit, [nand1, nand2, nand3])
    oscillatorRes.printDims()
    
    # Run oscillator res forward
    osc_inputs = np.zeros((3, 7000, 4))
    outputs = oscillatorRes.run4input(osc_inputs)
    plotters.InOutSplit(osc_inputs, outputs, "Oscillator") """


if __name__ == "__main__":
    main()