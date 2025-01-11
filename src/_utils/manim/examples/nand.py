from _prnn.reservoir import Reservoir
from _utils import inputs
from _utils.manim.inout_plt import CombinedInputOutputPlot
from manim import *
from manim import config


config.media_width = "75%"
config.frame_rate = 30
config.media_dir = "src/_utils/manim/examples/media"

# Circuit configuration and generation.
nand: Reservoir = Reservoir.load("nor")

time = 4000
inps = inputs.high_low_inputs(time)
name = "NOR Gate"
outputs = nand.run(inps)

scene = CombinedInputOutputPlot(time, outputs, inps, name)

scene.render()
