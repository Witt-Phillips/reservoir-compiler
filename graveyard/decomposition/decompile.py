from decomposition.universal_decomposition import *
from reservoir import Reservoir
import numpy as np
import re

def decompile(self: Reservoir, order):
    prefactors, bases, B_coefs, tanh_derivs = regex_parse_tseries(symbolic_tseries(self, len(self.x_init), order), verbose=False)
    # TODO: slice tseries matrix

    print("prefactors:", prefactors)
    print("bases:", bases)
    print("B_coefs:", B_coefs)
    print("tanh_derivs:", tanh_derivs)

    self.print()

    R = np.empty((0, len(bases)))
    for state in range(0, len(self.r_init)):
        coefs = []
        # get coef for given base
        for idx in range(0, len(bases)):
            coef = 1
            coef *= prefactors[idx]
            # multiply out B_coefs
            for B_coef in B_coefs[idx].free_symbols: # ignores 1 entry
                coef *= self.B[state][int(re.search(r'\d+', B_coef.name).group())]
            # multiply out tanh_derivs
            coef *= tanh_derivs[idx].subs('d', self.d[state])
            coefs.append(float(coef))
        R = np.vstack((R, np.array(coefs)))
    R = R * -1 # negate to match Jason - ask about this
    print("R:\n", np.round(R, 2))
    print("s:", bases)
    return R, bases

Reservoir.decompile = decompile