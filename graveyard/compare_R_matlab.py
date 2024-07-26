import numpy as np
# DEPRECATED
matlab_output = np.loadtxt('src/utils/comparison_outputs/RsNPL1_matlab.csv', delimiter=',')
python_output = np.loadtxt('src/utils/comparison_outputs/RsNPL1_python.csv', delimiter=',')

# check if python columns are present in both matrices
def compare_columns(matlab_matrix, python_matrix):
    for i in range(python_matrix.shape[1]):
        python_column = python_matrix[:, i]
        found = False
        for j in range(matlab_matrix.shape[1]):
            matlab_column = matlab_matrix[:, j]
            if np.allclose(python_column, matlab_column):
                found = True
                break
        if not found:
            print(f"Column {i} in Python output not found in MATLAB output.")
            return False
    return True

# Compare the columns
if compare_columns(matlab_output, python_output):
    print("All columns in the Python output are present in the MATLAB output.")
else:
    print("Some columns in the Python output are not present in the MATLAB output.")