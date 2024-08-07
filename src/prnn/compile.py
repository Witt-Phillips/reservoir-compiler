import numpy as np
import itertools

# generate the permutation matrix for partial derivatives
def partial_derivs(variables, max_order):
    combinations = []
    for order in range(max_order + 1):
        for combo in itertools.combinations_with_replacement(range(variables), order):
            # count of each variable
            count = [0] * variables
            for index in combo:
                count[index] += 1
            combinations.append(count)
    return np.array(combinations)


# problem is how many combinations of the three inputs exist?