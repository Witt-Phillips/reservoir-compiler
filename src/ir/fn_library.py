""" 
Library mapping opcodes to reservoirs, which are loaded from src/presets
"""

rnn_lib = {
    # name, (inputs, outputs, path)
    "NAND": (2, 1, "nand"),
    "NAND_DE": (2, 2, "nand_de"),
    "NAND_TRIPLE": (2, 3, "nand_triple"),
    "NOR": (2, 1, "nor"),
    "LORENZ": (0, 3, "lorenz"),
    "ROTATE90": (3, 3, "rotation90"),
}
