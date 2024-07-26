import numpy as np
import pandas as pd

def compare_matrices_with_tolerance(csv_path1, csv_path2, percent_tolerance, verbose=False):
    # Load matrices from CSV files
    matrix1 = pd.read_csv(csv_path1, header=None).values
    matrix2 = pd.read_csv(csv_path2, header=None).values
    
    # Ensure matrices have the same number of rows
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError("Matrices must have the same number of rows")
    
    # Convert percent tolerance to a decimal
    tolerance = percent_tolerance / 100.0
    
    # Keep track of which columns in matrix2 have been matched
    matched_columns = [False] * matrix2.shape[1]
    
    # Iterate over each column in matrix1
    for col1 in range(matrix1.shape[1]):
        col1_values = matrix1[:, col1]
        match_found = False
        
        # Iterate over each column in matrix2
        for col2 in range(matrix2.shape[1]):
            if matched_columns[col2]:
                continue
            
            col2_values = matrix2[:, col2]
            # Check if all values are within the tolerance
            if np.all(np.abs((col1_values - col2_values) / col1_values) <= tolerance):
                match_found = True
                matched_columns[col2] = True
                if verbose:
                    avg_percent_error = np.mean(np.abs((col1_values - col2_values) / col1_values)) * 100
                    print(f"Match found for column {col1 + 1} in matrix1 with column {col2 + 1} in matrix2.")
                    print(f"Column {col1 + 1} values (matrix1): {col1_values}")
                    print(f"Column {col2 + 1} values (matrix2): {col2_values}")
                    print(f"Average percent error: {avg_percent_error:.2f}%")
                break
        
        if not match_found:
            print(f"No match found for column {col1 + 1} in matrix1.")
            print(f"Column {col1 + 1} values: {col1_values}")
            return False
    
    print("All columns in matrix1 have a match in matrix2 within the given tolerance.")
    return True

def main():
    csv_path1 = "src/utils/comparison_outputs/C1_python.csv"
    csv_path2 = "src/utils/comparison_outputs/C1_matlab.csv"
    percent_tolerance = 5.0
    compare_matrices_with_tolerance(csv_path1, csv_path2, percent_tolerance, verbose=True)

if __name__ == "__main__":
    main()