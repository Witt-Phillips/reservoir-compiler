import sympy as sp
import sympy as sp
from reservoir import Reservoir


def regex_parse_tseries(tseries: sp.Expr, verbose=False):
    prefactors = []
    bases = []
    B_coefs = []
    tanh_derivs = []
    if verbose:
        print("terms:")
    for term in tseries.as_ordered_terms():
        x_term = 1
        B_term = sp.sympify(1)  
        non_x_B_term = sp.sympify(1)   
        prefactor = sp.sympify(1)   

        for factor in term.as_ordered_factors():
            # Prefactor
            if factor.is_constant():
                prefactor *= factor  
            # x symbols
            elif isinstance(factor, sp.Symbol) and 'x' in str(factor):
                x_term *= factor  
            elif isinstance(factor, sp.Pow) and 'x' in str(factor.base):
                x_term *= factor  
            # B-coefs
            elif isinstance(factor, sp.Symbol) and factor.name.startswith('B'):
                B_term *= factor  
            elif isinstance(factor, sp.Pow) and isinstance(factor.base, sp.Symbol) and factor.base.name.startswith('B'):
                B_term *= factor  
            else:
                non_x_B_term *= factor

        prefactors.append(prefactor)
        bases.append(x_term)
        B_coefs.append(B_term)
        tanh_derivs.append(non_x_B_term)
        if verbose:
            print(term)
    if verbose:
        print("prefactors:", prefactors)
        print("bases:", bases)
        print("B_coefs:", B_coefs)
        print("tanh_derivs:", tanh_derivs)
    return prefactors, bases, B_coefs, tanh_derivs


def symbolic_tseries(self: Reservoir, input_size, order):
    x_syms = sp.symbols('x:' + str(int(input_size)))
    xdot_syms = sp.symbols('xdot:' + str(int(input_size)))
    inputs = x_syms + xdot_syms
    
    B_syms = sp.symbols('B:' + str(input_size)) # each B_sym is a column of B
    d = sp.symbols('d')
    gam = sp.symbols('gam')

    # generate tanh needed
    y = sp.symbols('y') # placeholder var for diff
    g = sp.tanh(y)
    dg = g.diff(y)


    Bx = sum([b * x for b, x in zip(B_syms, x_syms)])
    Bxdot = sum([b * x for b, x in zip(B_syms[input_size:], xdot_syms)])
    h: sp.Expr = -g.subs(y, Bx + d) - ((1 / gam) * (dg.subs(y, Bx + d) * Bxdot)) # -g(Bx + d) - dg(Bx + d) * Bxdot    
    h_tay: sp.Expr = taylor_sympy(h, inputs, [0] * len(inputs), order)
    h_tay = h_tay.subs(gam, self.global_timescale)
    #h_tay.simplify()
    return h_tay

# from https://stackoverflow.com/questions/22857162/multivariate-taylor-approximation-in-sympy
# super duper slow
def taylor_sympy(function_expression, variable_list, evaluation_point, degree):
    """
    Mathematical formulation reference:
    https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/3%3A_Topics_in_Partial_Derivatives/Taylor__Polynomials_of_Functions_of_Two_Variables
    :param function_expression: Sympy expression of the function
    :param variable_list: list. All variables to be approximated (to be "Taylorized")
    :param evaluation_point: list. Coordinates, where the function will be expressed
    :param degree: int. Total degree of the Taylor polynomial
    :return: Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools

    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution

    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term

    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial