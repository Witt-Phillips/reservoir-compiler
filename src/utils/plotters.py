from reservoir import Reservoir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np    

def InOutSplit(inputs, outputs, title):
    if inputs.shape[2] > 1:
        inputs = inputs[:, :, 0]
        print("warning: plot: inputs must be 2D, taking first z-axis")

    if inputs.shape[1] != outputs.shape[1]:
        raise ValueError("plot: inputs and outputs must have equal length.")
    
    time = np.arange(inputs.shape[1])
    plt.figure(figsize=(12, 8))

    # signals
    plt.subplot(2, 1, 1)
    for i in range(inputs.shape[0]):
        plt.plot(time, inputs[i, :], label=f'Signal {i+1}')
    
    # Adding a 10% margin to y-limits for inputs
    inputs_min, inputs_max = np.min(inputs), np.max(inputs)
    inputs_margin = (inputs_max - inputs_min) * 0.1
    plt.ylim(inputs_min - inputs_margin, inputs_max + inputs_margin)

    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.title(f'{title} - Input(s)')
    plt.legend()

    # outputs 
    plt.subplot(2, 1, 2)
    for j in range(outputs.shape[0]):
        plt.plot(time, outputs[j, :], label=f'Output {j+1}')
    
    # Adding a 10% margin to y-limits for outputs
    outputs_min, outputs_max = np.min(outputs), np.max(outputs)
    outputs_margin = (outputs_max - outputs_min) * 0.1
    plt.ylim(outputs_min - outputs_margin, outputs_max + outputs_margin)

    plt.xlabel('Time')
    plt.ylabel('Output Value')
    plt.title(f'{title} - Output(s)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def Outputs(outputs, title):
    time = np.arange(outputs.shape[1])
    plt.figure(figsize=(12, 8))

    # Outputs only
    for j in range(outputs.shape[0]):
        plt.plot(time, outputs[j, :], label=f'Output {j+1}')
    
    # Adding a 10% margin to y-limits for outputs
    outputs_min, outputs_max = np.min(outputs), np.max(outputs)
    outputs_margin = (outputs_max - outputs_min) * 0.1
    plt.ylim(outputs_min - outputs_margin, outputs_max + outputs_margin)

    plt.xlabel('Time')
    plt.ylabel('Output Value')
    plt.title(f'{title}')
    plt.legend()

    plt.tight_layout()
    plt.show()

def threeD(outputs, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(outputs[0, :], outputs[1, :], outputs[2, :])
    ax.set_title(title)
    plt.show()

def threeDInputOutput(inputs, outputs, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(inputs[0, :], inputs[1, :], inputs[2, :], label='Input')
    ax.plot(outputs[0, :], outputs[1, :], outputs[2, :], label='Output')
    ax.set_title(title)
    plt.legend()
    plt.show()