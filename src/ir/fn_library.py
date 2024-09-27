""" 
Library mapping opcodes to reservoirs, which are loaded from src/presets
"""

rnn_lib = {
    # name, (inputs, outputs, path)
    "NAND": (2, 1, "nand"),
    "NOR": (2, 1, "nor"),
    "LORENZ": (1, 3, "lorenz"),
    "ROTATE90": (3, 3, "rotation90"),
}
