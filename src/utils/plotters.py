""" Various plotting functions for different network types """

import matplotlib.pyplot as plt
import numpy as np


def plot_matrix_heatmap(matrix: np.ndarray, title: str = "Matrix Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()


def plot_reservoir_matrices(reservoir, title_prefix="Reservoir Matrices"):
    """
    Plots all matrices and vectors in a reservoir in one big plot with subplots.

    Args:
    reservoir: A Reservoir object containing matrices A, B, W, and vectors x_init, r_init, and d.
    title_prefix: A string to prefix all plot titles.
    """
    components = {
        "A": reservoir.A,
        "B": reservoir.B,
        "W": reservoir.W,
        "x_init": reservoir.x_init,
        "r_init": reservoir.r_init,
        "d": reservoir.d,
    }

    num_components = sum(
        1
        for component in components.values()
        if component is not None and component.size > 0
    )

    # Calculate the number of rows and columns for the subplots
    num_cols = 2
    num_rows = (num_components + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2.5 * num_rows))

    if num_rows == 1:
        axes = [axes]

    idx = 0
    for name, component in components.items():
        if component is not None and component.size > 0:
            # Reshape vectors to 2D arrays for consistent plotting
            if component.ndim == 1:
                component = component.reshape(-1, 1)

            ax = axes[idx // num_cols, idx % num_cols]
            im = ax.imshow(component, cmap="viridis", aspect="auto")
            fig.colorbar(im, ax=ax, label="Value")
            ax.set_title(f"{title_prefix} - {name}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            idx += 1

    plt.tight_layout()
    plt.show()


def in_out_split(inputs, outputs, title):
    """2 plots; one with inputs, the other with outputs. Good for 2d"""
    if inputs.shape[1] != outputs.shape[1]:
        raise ValueError("plot: inputs and outputs must have equal length.")

    time = np.arange(inputs.shape[1])
    plt.figure(figsize=(12, 8))

    # signals
    plt.subplot(2, 1, 1)
    for i in range(inputs.shape[0]):
        plt.plot(time, inputs[i, :], label=f"Signal {i+1}")

    # Adding a 10% margin to y-limits for inputs
    inputs_min, inputs_max = np.min(inputs), np.max(inputs)
    inputs_margin = (inputs_max - inputs_min) * 0.1
    plt.ylim(inputs_min - inputs_margin, inputs_max + inputs_margin)

    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.title(f"{title} - Input(s)")
    plt.legend()

    # outputs
    plt.subplot(2, 1, 2)
    for j in range(outputs.shape[0]):
        plt.plot(time, outputs[j, :], label=f"Output {j+1}")

    # Adding a 10% margin to y-limits for outputs
    outputs_min, outputs_max = np.min(outputs), np.max(outputs)
    outputs_margin = (outputs_max - outputs_min) * 0.1
    plt.ylim(outputs_min - outputs_margin, outputs_max + outputs_margin)

    plt.xlabel("Time")
    plt.ylabel("Output Value")
    plt.title(f"{title} - Output(s)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plt_outputs(outputs, title, output_names):
    if outputs.shape[0] == 0:
        print("No outputs to plot")
        return

    """Basic plotter designed for output portions"""
    time = np.arange(outputs.shape[1])
    plt.figure(figsize=(12, 8))

    # outputs only
    for j in range(outputs.shape[0]):
        label = output_names[j] if j < len(output_names) else f"Output {j+1}"
        plt.plot(time, outputs[j, :], label=label)

    # Adding a 10% margin to y-limits for outputs
    outputs_min, outputs_max = np.min(outputs), np.max(outputs)
    outputs_margin = (outputs_max - outputs_min) * 0.1
    plt.ylim(outputs_min - outputs_margin, outputs_max + outputs_margin)

    plt.xlabel("Time")
    plt.ylabel("Output Value")
    plt.title(f"{title}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def three_d(outputs, title):
    """3d plotter for dynam sys like Lorenz"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(outputs[0, :], outputs[1, :], outputs[2, :])
    ax.set_title(title)
    plt.show()


def three_d_input_output(inputs, outputs, title):
    """3D plotter that also shows input space (3d)"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(inputs[0, :], inputs[1, :], inputs[2, :], label="Input")
    ax.plot(outputs[0, :], outputs[1, :], outputs[2, :], label="Output")
    ax.set_title(title)
    plt.legend()
    plt.show()
