from pyres import std


def main(i1, i2, i3):
    i4 = i1 and i2
    o1 = std.nand(i4, i3)
    return o1
