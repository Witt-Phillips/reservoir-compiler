from reservoir import *
import sympy as sp
from prnn import equations
from utils import inputs

o1, o2, o3, s1, s2, s3 = sp.symbols('o1 o2 o3 s1 s2 s3')
test_eqs = [
    sp.Eq(o1, -s2 + o1),
    sp.Eq(o2, o1 - s1),
    sp.Eq(o3, s3)
]
if 0:
    for eq in test_eqs:
        print(sp.octave_code(eq))

eqs, recurrences = equations.process(test_eqs)

print(eqs)
print(recurrences)
