import numpy as np
from pyres import compile
from _utils.plotters import plt_outputs

res = compile("examples/frontend/src_code/ex1.pyres")

inp = np.zeros((1, 4000))
outputs = res.run(inp)
plt_outputs(outputs, "Nand Series", res.output_names)