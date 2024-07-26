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

    currentlyRunning = 'and'
    
    # list of symbolic output equations
    logic_eqs = [
        sp.Eq(o1, pitchfork_bifurcation + logic[currentlyRunning])
    ]
    
    # display octave code
    if verbose:
        for eq in logic_eqs:
            print(sp.octave_code(eq))

    # generate inputs
    logic_pt = inputs.high_low_inputs(1000)

    # solve for W/ internalize recurrencies
    nand_res, W, _ = runMethod(logic_eqs, logic_pt)
    nand_res: Reservoir

    # run network forward
    outputs = nand_res.run4input(W, logic_pt)
    plotters.InOutSplit(logic_pt, outputs, currentlyRunning + " Gate")

    # Circuit configuration and generation.
    nand1 = nand_res.copy()
    nand2 = nand_res.copy()
    nand3 = nand_res.copy()

    nand1.W = np.random.rand(1, 5)
    nand2.W = np.random.rand(1, 5)
    nand3.W = np.random.rand(1, 5)

    # sets the input number of net2 to the output number of net1
    # (outputNet, o#, inputNet, x#)
    oscillator_circuit = [
        [nand1, 1, nand2, 1],
        [nand1, 1, nand2, 2],
        [nand2, 1, nand3, 1],
        [nand2, 1, nand3, 2],
        [nand3, 1, nand1, 1],
        [nand3, 1, nand1, 2]
    ]
    
    # TODO: keep track of which input remain exposed
    # retain B accordingly
    # concat rs
    # determine xs lenght (confirmed is 0 if na)
    #oscillator_reservoir = circuit.connect(oscillator_circuit, [nand1, nand2, nand3])

if __name__ == "__main__":
    main()