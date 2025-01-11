from pyres import compile
from _utils.plotters import plt_outputs, three_d, in_out_split
from _utils.inputs import high_low_inputs
import numpy as np

path = "examples/frontend/src_code/internalize_c.pyres"
R = compile(path, verbose=False)
t = 10000
# input = np.full((1, t), 0.1)
# input = high_low_inputs(t)
o = R.run(time=t)

plt_outputs(R, o, path)
# three_d(R, o, path)
# in_out_split(R, input, o, path)
