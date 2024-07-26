import matlab.engine
import numpy as np

# converts a numpy double array into a numpy array
def matrix_mat2np(mat_matrix: 'matlab.double'):
    if isinstance(mat_matrix, matlab.double):
        matrix = [[float(element) for element in row] for row in mat_matrix]
        return np.array(matrix)
    else:
        raise TypeError("Input must be a matlab.double type")