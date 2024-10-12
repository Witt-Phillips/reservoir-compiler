""" Graph class; target of IR core compiler """

import networkx as nx
import matplotlib.pyplot as plt
from _prnn.reservoir import Reservoir


class CGraph:
    """
    Class to represent a computational graph for reservoirs.
    Handles nodes for inputs, outputs, variables, and reservoirs,
    and maintains edges with input/output indices for reservoir channels.
    """

    def __init__(self):
        """Initialize an empty directed graph"""
        self.graph = nx.DiGraph()

    def add_node(self, name: str, node_type: str, **attrs):
        """
        General method fto add a node to the graph with a specified type.
        Args:
            name (str): The node's name.
            node_type (str): The type of node ('input', 'var', 'reservoir', 'output').
            **attrs: Additional attributes for the node.
        """
        # assert not self.graph.has_node(name), f"Node {name} already exists"
        if not self.graph.has_node(name):
            self.graph.add_node(name, type=node_type, **attrs)
        else:
            current_attrs = self.graph.nodes[name]  # Get current attributes
            current_attrs.update(attrs)  # Update with new attrs
            current_attrs["type"] = node_type

    def get_node(self, name: str):
        """
        Returns the node with the given name.
        """
        return self.graph.nodes[name]

    def get_var_source(self, name: str) -> tuple[Reservoir, int]:
        """
        Returns the source reservoir and its output index for a variable node.
        Ensures that there is only one predecessor.
        """
        predecessors = list(self.graph.predecessors(name))
        if len(predecessors) != 1:
            raise ValueError(
                f"Variable {name} should have exactly one source, but has {len(predecessors)}"
            )
        source = predecessors[0]
        edge_data = self.graph.get_edge_data(source, name)
        return (self.graph.nodes[source]["reservoir"], edge_data["output_idx"])

    def get_var_target(self, name: str) -> tuple[Reservoir, int]:
        """
        Returns the target reservoir and its input index for a variable node.
        Ensures that there is only one successor.
        """
        successors = list(self.graph.successors(name))
        if len(successors) != 1:
            raise ValueError(
                f"Variable {name} should have exactly one target, but has {len(successors)}"
            )
        target = successors[0]
        edge_data = self.graph.get_edge_data(name, target)
        return (self.graph.nodes[target]["reservoir"], edge_data["input_idx"])

    def all_nodes(self):
        """
        Returns all nodes in the graph.
        """
        return self.graph.nodes

    def all_edges(self):
        """
        Returns all edges in the graph.
        """
        return self.graph.edges

    def add_input(self, name: str, val: float = None):
        """Adds an input node to the graph."""
        self.add_node(name, "input", value=val)

    def make_return(self, name: str):
        """
        Converts an existing node to a return node.
        - Checks that the node exists and is not a reservoir.
        - Ensures the node has no outputs.
        """
        assert self.graph.has_node(name), f"Node {name} does not exist"
        assert (
            self.graph.nodes[name]["type"] != "reservoir"
        ), f"Cannot convert reservoir {name} to return"
        assert (
            self.graph.out_degree(name) == 0
        ), f"Node {name} already has outputs, cannot convert to return"

        # Change the node's type to 'return' and ensure it has one input and no outputs
        self.graph.nodes[name]["type"] = "return"

    def add_var(self, name: str):
        """Adds a variable node to the graph."""
        self.add_node(name, "var")

    def add_reservoir(self, name: str, reservoir: Reservoir):
        """
        Adds a reservoir node to the graph.
        Args:
            name (str): The reservoir's name.
            reservoir (Reservoir): The reservoir object associated with this node.
        """
        self.add_node(name, "reservoir", reservoir=reservoir)

    def add_output(self, name: str):
        """Adds an output node to the graph."""
        # check if overwriting input; if so, give _out tag
        node = self.get_node(name)
        if node:
            if node["type"] == "var":
                self.make_return(name)
                return
            elif node["type"] == "input":
                name = name + "_out"

        self.add_node(name, "output")

    def add_edge(
        self, source: str, target: str, out_idx: int = None, in_idx: int = None
    ):
        """
        Adds an edge between two nodes in the graph, carrying input/output indices when needed.
        Args:
            source (str): The source node.
            target (str): The target node.
            out_idx (int, optional): The output index if the source is a reservoir.
            in_idx (int, optional): The input index if the target is a reservoir.
        """
        assert self.graph.has_node(source), f"Source node {source} does not exist"
        assert self.graph.has_node(target), f"Target node {target} does not exist"

        if self.graph.nodes[source]["type"] == "reservoir":
            assert (
                out_idx is not None
            ), "output_idx is required when source is a reservoir"
        else:
            out_idx = None  # Clear output_idx for non-reservoir sources

        if self.graph.nodes[target]["type"] == "reservoir":
            assert (
                in_idx is not None
            ), "input_idx is required when target is a reservoir"
        else:
            in_idx = None  # Clear input_idx for non-reservoir targets

        self.graph.add_edge(source, target, output_idx=out_idx, input_idx=in_idx)

    def get_graph(self):
        """
        Returns the underlying NetworkX graph.
        """
        return self.graph

    def print(self):
        """
        Prints all nodes and edges in the graph for debugging.
        """
        print("Graph nodes:")
        for node, data in self.graph.nodes(data=True):
            print(f"{node}: {data}")
        print("\nGraph edges:")
        for source, target, data in self.graph.edges(data=True):
            print(
                f"{source} -> {target}, output_idx: {data.get('output_idx')}, \
                    input_idx: {data.get('input_idx')}"
            )

    def is_directed(self):
        """
        Returns True if the graph is directed, False otherwise.
        """
        return self.graph.is_directed()

    def draw(self):
        # Define node and edge colors
        node_color = "lightblue"
        edge_color = "gray"
        node_border_color = "white"
        font_color = "darkblue"
        edge_width = 2

        # Position nodes using the basic spring layout
        pos = nx.spring_layout(self.graph)

        # Define colors for different node types
        node_colors = []
        for node in self.graph.nodes(data=True):
            if node[1]["type"] == "input":
                node_colors.append("lightgreen")
            elif node[1]["type"] == "return":
                node_colors.append("lightcoral")
            elif node[1]["type"] == "reservoir":
                node_colors.append("lightblue")
            else:
                node_colors.append("skyblue")  # Default color

        # Draw nodes with a professional look and ensure proper spacing
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=2500,  # Increase node size to hold text
            node_color=node_colors,  # Use the defined colors for nodes
            edge_color="black",  # Darker edges for better contrast
            font_size=8,  # Standard font size for readability
            font_weight="regular",  # Regular font weight for labels
            font_color="black",  # Font color for readability
            linewidths=1.5,  # Standard borders for nodes
            edgecolors="black",  # Node border color
        )

        # Customize edge widths and transparency
        nx.draw_networkx_edges(
            self.graph,
            pos,
            width=1.5,  # Standard edge width for better visibility
            alpha=0.9,  # Slight transparency for edges
        )

        # Add a title and adjust layout for better spacing
        plt.title("Graph Visualization", fontsize=14)
        plt.tight_layout(pad=2)  # Adjust layout for better spacing
        plt.show()
