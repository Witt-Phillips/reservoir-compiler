from _prnn.reservoir import Reservoir
from _utils import inputs
from _utils.manim.plt_outputs3d import ResOutputPlotThreeD
from manim import *
from manim import config


config.media_width = "75%"
config.frame_rate = 30
config.media_dir = "src/_utils/manim/examples/media"

# Circuit configuration and generation.
nand: Reservoir = Reservoir.load("lorenz")

time = 500
name = "lorenz"
outputs = nand.run(time=time)

# scene = ResOutputPlotThreeD(outputs, name)
scene = ResOutputPlotThreeD(outputs, name)
scene.render()
