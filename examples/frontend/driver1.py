from pyres import compile
from _utils.plotters import plt_outputs
from _utils.inputs import high_low_inputs, high_low_inputs_3rows

res = compile("examples/frontend/src_code/ex1.pyres", verbose=True)

if 1:
    # inp = high_low_inputs(4000)[0, :].reshape(1, -1)
    # print(inp)
    inp = high_low_inputs(4000)
    # inp = np.full((1, 4000), 0.5)
    # outputs = res.run(inp)
    outputs = res.run(inp)
    plt_outputs(outputs, "", res.output_names)
