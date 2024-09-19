<p align="center">
  <img src="rnn.svg" alt="RNN Graphic, Image based on Kim paper linked below." style="width: 50%;" />
</p>

# Framework for the Reservoir Compiler

Working towards a compiler to express high-level code in RNNs. Currently building a framework for programming RNNs based on Jason [Kim's method](https://www.nature.com/articles/s42256-023-00668-8).

## Table of Contents

- [Quick Start](#quick-start)
- [Examples](#examples)
- [Programming an RNN](#programming-an-rnn)
- [Circuit Design](#circuit-design)
- [Presets](#presets)

## Quick Start

Welcome to the framework for the reservoir compiler. If you do stuff like this a lot, just clone, set up a venv, install dependencies @ `requirements.txt`, and ensure you have Matlab installed & licensend. Then feel free to skip to examples.

### Basics

If not, all good! These commands accomplish what I just described.

```{bash}
$ git clone https://github.com/Witt-Phillips/reservoir-compiler
$ python -m venv venv
$ source myenv/bin/activate
$ pip install -r requirements.txt
```

### Verify Matlab

The current version of this framework uses the Matlab engine, but requires that you have Matlab installed on your machine and that you have an active license.

## Examples

All of the files in the examples folder (unless marked in their header as a work in progress) are executable. Each sets up an RNN or circuit of RNNs to accomplish a task. `logic_gates.py`, for example, offers a simple introduction to interpretting dynamical systems as logic.

## Programming an RNN

To program an RNN, we use the SymPy symbolic library to generate systems of equations. These equations must have only one symbol on the left-hand side, but are otherwise unconstrained in their naming and format.

A possible system of equations:

```{python}
logic_eqs = [
    sp.Eq(o1, -s2),
    sp.Eq(o2, s1),
    sp.Eq(o3, s3)
]
```

Given a system, use the `reservoir.solve()` method to intialize a reservoir that instantiates the given system.

### Internalized Recurrencies

The `reservoir.solve()` method determines the appropriate dimensions for reservoir inputs and outputs given the system of equations. When equations are parsed, inputs are categorized as either `signal` or `recurrent` (a recurrent input is one set equal to an output). `recurrent` inputs are internalized into the adjacency `A`, and are thus excluded from the input space of the updated reservoir.

## Circuit Design

To combine reservoirs into circuits, we use a simple, list-based language of the following format:

`[Output Network, Output Number, Input Network, Input Number]`, which expresses the recurrency. `OutputNetwork.o[output_number] == InputNetwork.o[input_number]`.

The `circuit.connect()` method takes a list of these tuples and returns a reservoir that implements the programmed circuit.

### Signal vs. Internal Inputs/ Outputs

Much like the internalization of recurrencies in the PRNN programming example, the circuit programming method recognizinges which inputs/ outputs of the orignial system remain exposed to the user after connecting the circuit.

Note that while a single output can be proted inot multipl inputs, teh connect() method is constrained in that multiple outputs cannot be set equal to a single input. While we can use outputs multiple times, the circuit method internalizes (removes from output vector of final reservoir) any output that has been routed to at least one input.

### Tip: Expose internal inputs by 'doubling' an output

The circuit language is designed such that outputs which feed into inputs of other matrices are not considered outputs of the completed circuit reservoir. Of course, we may need to read out these 'internal' outputs. Doing so is simple: just define another output and set it equal to the internalized output: `o2 = o1`, for example.

Alternatively (and more efficiently), use the reservoir.doubleOutput(n) method, which appends a copy of the nth row of W to the existing reservoir.

## Presets

Solving for reservoirs remains computationally expensive. To avoid recalcuation of reused reservoirs, the reservoir.save() and corresponding reservoir.load() methods allow storage in a .rsvr file.
