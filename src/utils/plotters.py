from reservoir import Reservoir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plt_r_evolution(reservoir: Reservoir, num_steps: int):
    states = [reservoir.r.copy()]

    for _ in range(num_steps):
        x = np.random.randn(len(reservoir.x_init))
        r = reservoir.propagate(x)
        states.append(r.copy())
    
    states = np.array(states)
    
    plt.figure(figsize=(12, 8))
    for i in range(min(10, states.shape[1])):  # Plot the first 10 components or fewer if less than 10
        plt.plot(states[:, i], label=f'r[{i}]')
    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.title('Evolution of Reservoir States (Random)')
    plt.legend()
    plt.show()

def plt_decompilation(C1, C2, C3a, C3b, C4a, C4b, C4c):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    sns.heatmap(C1, annot=True, cmap="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("C1 Coefficients")

    sns.heatmap(C2[:, :, 0], annot=True, cmap="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("C2 Coefficients (slice 0)")

    sns.heatmap(C3a[:, :, 0], annot=True, cmap="viridis", ax=axes[0, 2])
    axes[0, 2].set_title("C3a Coefficients (slice 0)")

    sns.heatmap(C3b[:, :, 0], annot=True, cmap="viridis", ax=axes[1, 0])
    axes[1, 0].set_title("C3b Coefficients (slice 0)")

    sns.heatmap(C4a[:, :, 0], annot=True, cmap="viridis", ax=axes[1, 1])
    axes[1, 1].set_title("C4a Coefficients (slice 0)")

    sns.heatmap(C4b[:, :, 0], annot=True, cmap="viridis", ax=axes[1, 2])
    axes[1, 2].set_title("C4b Coefficients (slice 0)")

    sns.heatmap(C4c[:, :, 0], annot=True, cmap="viridis", ax=axes[2, 0])
    axes[2, 0].set_title("C4c Coefficients (slice 0)")

    plt.tight_layout()
    plt.show()

def plt_rsnpl1(RsNPL1, title="RsNPL1"):
    nrows, ncols = RsNPL1.shape
    plt.figure(figsize=(max(10, ncols / 2), max(8, nrows / 2)))
    sns.heatmap(RsNPL1, annot=False, cmap="viridis")
    plt.title(title)
    plt.show()