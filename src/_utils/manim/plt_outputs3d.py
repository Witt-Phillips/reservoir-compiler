from manim import *
import numpy as np


class ResOutputPlotThreeD(ThreeDScene):
    def __init__(self, outputs: np.ndarray, name: str, **kwargs):
        super().__init__(**kwargs)
        self.outputs = outputs
        self.name = name

    def construct(self):
        # Check if outputs have dimension 3
        if self.outputs.shape[0] != 3:
            raise ValueError("Outputs must have 3 rows (dimensions) for 3D plotting.")

        # Scale down the entire scene to prevent overlap with the title
        scale_factor = 0.8  # Adjust this value as needed

        # Set up 3D axes with exact min and max limits
        axes = ThreeDAxes(
            x_range=[np.min(self.outputs[0]), np.max(self.outputs[0])],
            y_range=[np.min(self.outputs[1]), np.max(self.outputs[1])],
            z_range=[np.min(self.outputs[2]), np.max(self.outputs[2])],
            x_length=7,
            y_length=7,
            z_length=7,
        ).scale(scale_factor)

        # Add title
        title = Text(self.name, font_size=48).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        # Add the axes to the scene
        self.add(axes)

        # Create the points for the curve
        points = np.array(
            [
                axes.coords_to_point(x, y, z)
                for x, y, z in zip(self.outputs[0], self.outputs[1], self.outputs[2])
            ]
        )

        # Create a ValueTracker to control the drawing of the curve over time
        t_tracker = ValueTracker(0)

        # Create an empty VMobject for the curve
        curve = VMobject()
        curve.set_color(BLUE)
        curve.set_stroke(width=2)

        # Define the updater function to draw the curve over time
        def update_curve(mob):
            t = t_tracker.get_value()
            total_points = len(points)
            index = int(t * (total_points - 1)) + 1  # Ensure at least one point
            mob.set_points_as_corners(points[:index])

        # Add the updater to the curve
        curve.add_updater(update_curve)

        # Add the curve to the scene
        self.add(curve)

        # Set initial camera position
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Begin camera rotation to complete one rotation over the animation duration
        rotation_rate = (
            TAU / 50
        )  # TAU = 2 * PI radians, so this completes one rotation in 10 seconds
        self.begin_ambient_camera_rotation(rate=rotation_rate)

        # Animate the curve drawing
        self.play(t_tracker.animate.set_value(1), run_time=6, rate_func=linear)

        # Stop camera rotation after animation
        self.stop_ambient_camera_rotation()

        # Remove the updater once animation is complete
        curve.remove_updater(update_curve)

        # Keep the final frame for a moment
        self.wait(2)
