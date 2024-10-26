# from _utils.manim.inout_plt import CombinedInputOutputPlot
from _utils.manim.plt_outputs3d import ResOutputPlotThreeD

from manim import *
from manim import config

from _prnn.reservoir import Reservoir
from pyres import compile

config.media_width = "75%"
config.frame_rate = 30
config.media_dir = "src/_utils/manim/examples/media"

res = compile("examples/frontend/src_code/new_oscillator.py", verbose=True)
outputs = res.run(time=4000)

scene = ResOutputPlotThreeD(outputs, "Oscillator", window_size=200)
scene.render()
