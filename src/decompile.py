import numpy as np
import sympy as sp
import itertools
from reservoir import Reservoir

# Taylor Series approximation of reservoir machine code to 'order.' Returns R (coefs), s (symbolic bases)
def decompile(self: Reservoir, order, verbose=False):
    if verbose:
        print("Decompiling reservoir...")
        self.print()

    # 0th order T-series
    R = np.tanh(self.d)[:, None]
    s = np.array([2])[:, None]
    
    # init input symbols
    sym_names = [f'x{i+1}' for i in range(len(self.x_init))] # names bases based on x*
    x_syms = sp.symbols(sym_names)
    if verbose:
        print("Generated input symbols: ", x_syms)
    
    # generate necessary tanh derivatives
    tanh_derivs = gen_tanh_derivs(order)
    if verbose:
        print("Generated tanh derivatives: ", tanh_derivs)

    for o in range(1, order + 1):
        # calculate prefactor
        pre = 1 / sp.factorial(o)

        #generate symbolic bases as sympy array
        bases_arr = gen_bases(self.x_init, order) # tuples of symbol combinations
        s = np.concatenate((s, np.array([sp.Mul(*combination) for combination in bases_arr])[:, None]), axis=0)

        # Find B-coefs for each state function and append to R
        for n in range(0, len(self.r_init)):
            # create equation, where n = r_n. dtanh(Bn1x1 + ... + Bnkxk + d_n)
            equation = tanh_derivs[o].subs('x', sum(weight * base for weight, base in zip(self.B[n], x_syms)) + self.d[n])
            #print(equation)
			
            # TODO: pickup here

    if verbose:
        print("Decompile complete.")
        print("R: ", R)
        print("s: ", s)

    return R, s



Reservoir.decompile = decompile

# given input vector x and order, returns array of all possible base combinations as arrays of sympy symbols
# gen_bases([1, 2, 3], 2) => [(x1, x1), (x1, x2), (x1, x3), (x2, x1), (x2, x2), (x2, x3), (x3, x1), (x3, x2), (x3, x3)]
def gen_bases(x, order):
    var_names = [f'x{i+1}' for i in range(len(x))]
    sp_vars = sp.symbols(var_names)
    combinations = itertools.product(sp_vars, repeat=order)
    return list(combinations)


# return the first n derivatives of tanh (array of sympy functions)
# tanh_derivs(2) => [tanh(x), cosh(x)**(-2), -2*tanh(x)/cosh(x)**2]
def gen_tanh_derivs(n):
    x = sp.symbols('x')
    tanh_x = sp.tanh(x)
    derivatives = [tanh_x]

    for i in range(1, n + 1):
        derivatives.append(sp.diff(derivatives[-1], x))

    return [sp.simplify(d) for d in derivatives]

