from reservoir import Reservoir
from misc.manual_decompile import gen_tanh_derivs
import sympy as sp
import numpy as np
import re

# Input: Reservoir, order
    # sym_tay: coefs matrix, strings of expressions
    # Just need to parse that input string, evaluate the expression for proper B-coefs and x-values
# Output: R, s

# Drawn from file; coefs matrix and sym_tay matrix 

def eval_sym_tay():
    x = np.array([1, 2, 3, 4, 5])
    symtay = ["B1B2x1diff(g(d), d, d)", "something else", "x3x2x1", "", ""]
    
    # searches each expression in symtay for x values, multiplies them together into x_vectors
    x_coefs = np.ones(len(symtay))
    for expression in range(0, len(symtay)):
        values = re.findall(r'x(\d+)', symtay[expression])
        for value in values:
            if int(value) > 0:
                x_coefs[expression] *= float(x[int(value) - 1])
    print(x_coefs)

    # generate proper derivative #s
    derivs = np.zeros(len(symtay))

    for index in range(0, len(symtay)):
        expression = symtay[index]
        pattern = r'diff\(g\(d\),\s*((?:d,\s*)+)'
        match = re.search(pattern, expression)
        if match:
            derivs[index] = len(match.group().split(', ')) - 1
        else:
            derivs[index] = 0
    print(derivs)

    # grab B columns
    B_coefs = []
    for expression in range(0, len(symtay)):
        values = re.findall(r'B(\d+)', symtay[expression])
        B_coefs.append(values)            
    print(B_coefs)
    

def decompilev2(self: Reservoir, order, verbose=False):
    print("Hello world!")
    pass

Reservoir.decompilev2 = decompilev2
