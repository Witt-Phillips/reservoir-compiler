sim_lib = [
    ("AND", 2, 1, lambda a, b: a * b),                  
    ("OR", 2, 1, lambda a, b: max(a, b)),             
    ("NAND", 2, 1, lambda a, b: 1.0 - (a * b)),         
    ("NOR", 2, 1, lambda a, b: 1.0 - max(a, b)),        
    ("XOR", 2, 1, lambda a, b: a + b - 2 * a * b),      
    ("XNOR", 2, 1, lambda a, b: 1.0 - (a + b - 2 * a * b)),
    ("NOT", 1, 1, lambda a: 1.0 - a)         
]

rnn_lib = {
     # name, (inputs, outputs, path)
     'NAND': (2, 1, "nand"),
     'NOR': (2, 1, "nor"),
     'LORENZ': (1, 3, "lorenz"),
     'ROTATE90': (3, 3, "rotation90")
}