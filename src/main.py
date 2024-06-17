from reservoir import *
import matplotlib.pyplot as plt

# Example usage
def main():
    n = 5
    k = 3
    A = np.random.randn(n, n)
    B = np.random.randn(n, k)
    r_init = np.random.randn(n)
    x_init = np.random.randn(k)
    global_timescale = 0.1
    local_timescale = 1.0
    rs = np.random.randn(n)
    dv = np.random.randn(n)
    gam = 1.0
    o = 4

    # Clip r_init values to avoid invalid arctanh
    r_init = np.clip(r_init, -0.999999, 0.999999)

    # Instantiate the reservoir
    reservoir = Reservoir(A, B, r_init, x_init, global_timescale, local_timescale)
    
    # Run the reservoir forward one time step with a random input
    x = np.random.randn(k)
    reservoir.propagate(x)
    
    # Print the new state
    print("New state:", reservoir.r)

    # Perform decomposition
    Pd1, C1, C2, C3a, C3b, C4a, C4b, C4c = reservoir.decompose(rs, dv, gam, o)

    print("Pd1:", Pd1)
    print("C1:", C1)
    print("C2:", C2)
    print("C3a:", C3a)
    print("C3b:", C3b)
    print("C4a:", C4a)
    print("C4b:", C4b)
    print("C4c:", C4c)
    

def plot_drive(reservoir: Reservoir, num_steps: int):
    # Store the initial state
    states = [reservoir.r.copy()]

    # Run the reservoir forward multiple time steps with random inputs
    for _ in range(num_steps):
        x = np.random.randn(k)
        r = reservoir.propagate(x)
        states.append(r.copy())
    
    # Convert states to a NumPy array for easier slicing
    states = np.array(states)
    
    # Plot the evolution of the first few components of r
    plt.figure(figsize=(12, 8))
    for i in range(min(10, states.shape[1])):  # Plot the first 10 components or fewer if less than 10
        plt.plot(states[:, i], label=f'r[{i}]')
    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.title('Evolution of Reservoir States (Random)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
