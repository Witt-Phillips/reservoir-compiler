# src
# TODO: make it so that only the functions that main calls are compiled.
from pyres import std


def main():
    n3o1 = n3o2 = None
    n1o1, n1o2, n1go = std.nor3(n3o1, n3o2)
    n2o1, n2o2, n2go = std.nor3(n1o1, n1o2)
    n3o1, n3o2, n3go = std.nor3(n2o1, n2o2)
    return n1go, n2go, n3go


# driver
from pyres import compile
from _utils.plotters import plt_outputs


def driver():
    res = compile(__file__)
    outputs = res.run(time=4000)
    plt_outputs(outputs, "Oscillator", res.output_names)
