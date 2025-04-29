from graphviz import Digraph
import json


def visualize_gosdt_tree(tree, feature_names):
    """Visualize GOSDT Decision Tree using Graphviz."""
    dot = Digraph(comment="GOSDT Decision Tree")

    node_counter = 0  # Global counter to ensure unique node IDs

    def add_nodes_edges(node, parent=None, edge_label=""):
        """Recursive function to traverse the tree and add nodes/edges."""
        nonlocal node_counter
        node_id = node_counter
        node_counter += 1

        if "feature" in node:  # Internal node
            feature_label = f"{feature_names[node['feature']]}" if node[
                                                                       'feature'] < len(
                feature_names) else f"Feature {node['feature']}"
            label = f"{feature_label}?"
            dot.node(str(node_id), label)

            if parent is not None:
                dot.edge(str(parent), str(node_id), label=edge_label)

            add_nodes_edges(node["false"], node_id, "No")  # Left (False) branch
            add_nodes_edges(node["true"], node_id, "Yes")  # Right (True) branch

        else:  # Leaf node
            label = f"Class: {node['prediction']}\nLoss: {node['loss']:.4f}"
            dot.node(str(node_id), label, shape="box")

            if parent is not None:
                dot.edge(str(parent), str(node_id), label=edge_label)

    # Start visualization from the root node
    add_nodes_edges(tree[0])  # Root node is the first element in the list
    return dot
