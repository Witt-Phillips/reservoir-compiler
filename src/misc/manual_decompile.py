import numpy as np
import sympy as sp
import itertools
from reservoir import Reservoir

# Taylor Series approximation of reservoir machine code to 'order.' Returns R (coefs), s (symbolic bases)
# Doesn't account for interaction terms! (Decided to shelf this file in favor of decompilev2.py, as approach was clearly inefficient)

# Taylor Series approximation of reservoir machine code to 'order.' Returns R (coefs), s (symbolic bases)
def manual_decompile(self: Reservoir, order, verbose=False):
    if verbose:
        print("Decompiling reservoir...")
        self.print()

    # 0th order tseries to init R, s
    R = np.tanh(self.d)[:, None]
    s = np.array([1])[:, None]
    
    # init input symbols
    sym_names = [f'x{i+1}' for i in range(len(self.x_init))] # names bases based on x*
    x_syms = sp.symbols(sym_names)
    if verbose:
        print("Input symbols: ", x_syms)
    
    # generate necessary tanh derivatives
    tanh_derivs = gen_tanh_derivs(order)
    if verbose:
        print("Tanh derivatives: ", tanh_derivs)
        
    # calculates each order of the Taylor series, appends to R,s
    for o in range(1, order + 1):
        if verbose:
            print("----- ORDER", o, "------")
        # find prefactor
        pre = 1 / sp.factorial(o)
        if verbose:
            print("Prefactor: ", pre)

        #generate symbolic bases as sympy array
        bases_arr = gen_bases(self.x_init, o) # tuples of symbol combinations
        s = np.concatenate((s, np.array([sp.Mul(*combination) for combination in bases_arr])[:, None]), axis=0)

        # Find B-coefs for each state function and append to R
        R_section = np.empty((0, len(bases_arr)))
        for n in range(0, len(self.r_init)):
            # create equation where n = r_n. dtanh(Bn1x1 + ... + Bnkxk + d_n)
            base_equation = tanh_derivs[o].subs('x', sum(weight * base for weight, base in zip(self.B[n], x_syms)) + self.d[n])
            if verbose:
                print("(sanity check) base equation for neuron", n, ": ", base_equation)
			
            # taylor series for each base combination
            # TODO: not accounting for multiplicity of bases
            tseries = sp.S(0) 
            for base_combo in bases_arr:
                B_coefs = sp.S(1) 
                tseries_bases = sp.S(1)
                for base in base_combo:
                    idx = int(base.name[1]) - 1 # Extract the index from the sympy symbol. Bit clunky.
                    B_coefs *= self.B[n][idx]
                    tseries_bases *= base
                evaled_base_equation = base_equation.subs([(x_syms[i], self.x_init[i]) for i in range(len(self.x_init))])
                tseries += ((B_coefs * evaled_base_equation) * pre) * tseries_bases
            
            # Decompose tseries into R, s
            coefficients = []
            for term in tseries.as_ordered_terms():
                # extract coefficient and symbol from each term
                coeff, _ = term.as_coeff_Mul()
                coefficients.append(float(coeff))
            coefficients = np.array(coefficients)[None, :]

            if verbose:
                print("R section", R_section)
                print("Tseries for neuron", n, ": ", tseries)

            R_section = np.vstack((R_section, coefficients))
            
            

        R = np.hstack((R, R_section))


    if verbose:
        print("--------------------\nDecompile complete.")
        print("R:\n", R)
        print("s:\n", s)
    return R, s

# given input vector x and order, returns array of all possible base combinations as arrays of sympy symbols
# ex. gen_bases([1, 2, 3], 2) => [(x1, x1), (x1, x2), (x1, x3), (x2, x1), (x2, x2), (x2, x3), (x3, x1), (x3, x2), (x3, x3)]
def gen_bases(x, order):
    var_names = [f'x{I+1}' for I in range(len(x))]
    sp_vars = sp.symbols(var_names)
    combinations = itertools.combinations_with_replacement(sp_vars, order)
    return list(combinations)


# return the first n derivatives of tanh (array of sympy functions)
# ex. tanh_derivs(2) => [tanh(x), cosh(x)**(-2), -2*tanh(x)/cosh(x)**2]
def gen_tanh_derivs(n):
    x = sp.symbols('x')
    tanh_x = sp.tanh(x)
    derivatives = [tanh_x]

    for i in range(1, n + 1):
        derivatives.append(sp.diff(derivatives[-1], x))

    return [sp.simplify(d) for d in derivatives]