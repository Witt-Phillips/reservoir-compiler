from manim import *
import numpy as np


class PlotScene(Scene):
    def create_legend(self, labels, colors, position):
        legend_items = VGroup()
        for label, color in zip(labels, colors):
            line = Line(color=color, stroke_width=4).scale(0.5)
            text = Text(label, font_size=20, color=color).next_to(line, RIGHT, buff=0.2)
            legend_items.add(VGroup(line, text))

        legend_items.arrange(DOWN, aligned_edge=LEFT).next_to(position, RIGHT, buff=0.2)
        self.add(legend_items)

    def plot_outputs(self, axes_position, outputs, time):
        # Create axes for the outputs
        axes = Axes(
            x_range=[0, time, time / 10],
            y_range=[
                np.min(outputs) - 0.03,
                np.max(outputs) + 0.03,
                (np.max(outputs) - np.min(outputs)) / 10,
            ],
            axis_config={"color": GRAY_A},
            x_length=8,  # Scale down the x-axis
            y_length=2.5,  # Scale down the y-axis
            tips=False,
        ).move_to(
            axes_position + DOWN * 0.7
        )  # Move the graph down slightly

        self.add(axes)

        # Add x-axis label
        x_label = Text("t", font_size=24).next_to(axes.x_axis, LEFT)
        self.add(x_label)

        # Define colors
        colors = [RED, GREEN]
        x_values = np.linspace(0, time, num=outputs.shape[1])
        animations = []
        graphs = []
        labels = []
        colors_used = []

        # Plot each output row
        for i in range(outputs.shape[0]):
            y_values = outputs[i]
            color = colors[i % len(colors)]
            colors_used.append(color)
            output_graph = axes.plot_line_graph(
                x_values,
                y_values,
                add_vertex_dots=False,
                line_color=color,
                stroke_width=2,
            )
            graphs.append(output_graph)
            labels.append(f"o{i+1}")
            # Add the graph to animations
            animations.append(Create(output_graph))

        # Create a legend
        colors = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE]
        labels = [f"o{i+1}" for i in range(outputs.shape[0])]
        self.create_legend(labels, colors, position=axes.get_corner(UR))
        return animations

    def plot_inputs(self, axes_position, inputs, time):
        # Create axes for the inputs
        axes = Axes(
            x_range=[0, time, time / 10],
            y_range=[
                np.min(inputs) - 0.03,
                np.max(inputs) + 0.03,
                (np.max(inputs) - np.min(inputs)) / 10,
            ],
            axis_config={"color": GRAY_A},
            x_length=8,  # Scale down the x-axis
            y_length=2.5,  # Scale down the y-axis
            tips=False,
        ).move_to(
            axes_position
        )  # Move the graph down slightly
        self.add(axes)

        # Add x-axis label
        x_label = Text("t", font_size=24).next_to(axes.x_axis, LEFT)
        self.add(x_label)

        # Define colors
        colors = [BLUE, ORANGE, YELLOW, PURPLE]
        x_values = np.linspace(0, time, num=inputs.shape[1])
        animations = []
        graphs = []
        labels = []
        colors_used = []

        # Plot each input row
        for i in range(inputs.shape[0]):
            y_values = inputs[i]
            color = colors[i % len(colors)]
            colors_used.append(color)
            input_graph = axes.plot_line_graph(
                x_values,
                y_values,
                add_vertex_dots=False,
                line_color=color,
                stroke_width=2,
            )
            graphs.append(input_graph)
            labels.append(f"s{i+1}")
            # Add the graph to animations
            animations.append(Create(input_graph))

        # Create a legend
        labels = [f"s{i+1}" for i in range(inputs.shape[0])]
        self.create_legend(labels, colors, position=axes.get_corner(UR))

        return animations


class OnlyOutputPlot(PlotScene):
    def __init__(self, time: int, outputs: np.ndarray, name: str, **kwargs):
        super().__init__(**kwargs)
        self.time = time
        self.outputs = outputs
        self.name = name

    def construct(self):
        # Add title
        title = Text(self.name, font_size=48).to_edge(UP)
        self.add(title)

        # Plot outputs in the center
        animations = self.plot_outputs(
            axes_position=ORIGIN, outputs=self.outputs, time=self.time
        )

        # Play the output animation
        self.play(AnimationGroup(*animations, lag_ratio=0), run_time=4)
        self.wait()


class OnlyInputPlot(PlotScene):
    def __init__(self, time: int, inputs: np.ndarray, name: str, **kwargs):
        super().__init__(**kwargs)
        self.time = time
        self.inputs = inputs
        self.name = name

    def construct(self):
        # Add title
        title = Text(self.name, font_size=48).to_edge(UP)
        self.add(title)

        # Plot inputs in the center
        animations = self.plot_inputs(
            axes_position=ORIGIN, inputs=self.inputs, time=self.time
        )

        # Play the input animation
        self.play(AnimationGroup(*animations, lag_ratio=0), run_time=4)
        self.wait()


class CombinedInputOutputPlot(PlotScene):
    def __init__(
        self, time: int, outputs: np.ndarray, inputs: np.ndarray, name: str, **kwargs
    ):
        super().__init__(**kwargs)
        self.time = time
        self.outputs = outputs
        self.inputs = inputs
        self.name = name

    def construct(self):
        # Add title
        title = Text(self.name, font_size=48).to_edge(UP)
        self.add(title)

        # Plot outputs on the top part
        output_animations = self.plot_outputs(
            axes_position=UP * 2, outputs=self.outputs, time=self.time
        )

        # Plot inputs on the bottom part
        input_animations = self.plot_inputs(
            axes_position=DOWN * 2, inputs=self.inputs, time=self.time
        )

        # Play both animations simultaneously
        self.play(
            AnimationGroup(*output_animations, *input_animations, lag_ratio=0),
            run_time=4,
        )
        self.wait()
