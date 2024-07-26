from utils import utils, plotters
import numpy as np

inputs = utils.high_low_inputs(1000)
outputs = np.random.rand(1, 4000)

plotters.InOutSplit(inputs, outputs, "tester")