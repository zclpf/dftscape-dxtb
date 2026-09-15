"""
Visualization helpers for CriticalPointGraph.

Provides classes and functions to visualize the graph structure,
particularly for hierarchical/top-down layouts.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap
import py3Dmol


def visualize_xyz_py3dmol(xyz_string, label=None):
    """Visualize molecule from XYZ string using py3Dmol"""
    view = py3Dmol.view(width=600, height=400)
    view.addModel(xyz_string, "xyz")
    if label:
        view.addLabel(
            label,
            {
                "position": {"x": 0, "y": -3, "z": 0},
                "backgroundColor": "white",
                "fontColor": "black",
            },
        )
    view.setStyle({"stick": {}})
    view.setStyle(
        {'elem':'C'}, {'stick':{'color':'darkgrey'}}
    )
    view.setStyle(
        {'elem':'H'}, {'stick':{'color':'lightcyan'}}
    )
    view.setStyle(
        {'elem':'O'}, {'stick':{'color':'red'}}
    )
    view.zoomTo()
    return view


class CriticalPointGraphVisualizer:
    """
    Visualizer for CriticalPointGraph using matplotlib and NetworkX.

    Provides methods to plot the graph in various layouts, including
    hierarchical top-down arrangements.
    """

    def __init__(self, graph):
        """
        Initialize with a CriticalPointGraph or NetworkX graph.

        Args:
            graph: SearchableCriticalPointGraph or nx.Graph
        """
        self.graph = graph
        self.nx_graph = getattr(graph, "_graph", graph)

    @staticmethod
    def _node_attr_safe(nx_graph, node, attr):
        """
        Safely access a node attribute from the graph.

        Handles nested data structures (dict or object).

        Args:
            nx_graph: NetworkX graph
            node: Node identifier
            attr: Attribute name

        Returns:
            Attribute value or None if not found
        """
        ndata = nx_graph.nodes[node]
        obj = ndata.get("data", ndata) if isinstance(ndata, dict) else ndata
        if isinstance(obj, dict):
            return obj.get(attr, None)
        if hasattr(obj, attr):
            return getattr(obj, attr)
        try:
            return obj[attr]
        except Exception:
            return None

    def plot_hierarchy(
        self,
        layer_attr="index",
        color_attr="index",
        show_labels=True,
        max_labels=200,
        figsize=(10, 6),
        cmap="viridis",
        state_label=False,
        label_function=lambda x: f"\n{x.energy:.3f}"
    ):
        """
        Plot the graph in a top-down hierarchical layout.

        Nodes are arranged in layers by the layer_attr (top = largest value),
        with edges connecting them. Nodes are colored by color_attr.

        Args:
            layer_attr: Node attribute for layering (str)
            color_attr: Node attribute for coloring (str)
            show_labels: Whether to show node labels (bool)
            max_labels: Max nodes to label (int)
            figsize: Figure size (tuple)
            cmap: Colormap name (str)
            state_label: Whether to show cp.state next to nodes (bool)
        """
        nodes = list(self.nx_graph.nodes())
        if not nodes:
            print("Graph is empty")
            return

        # Compute numeric levels for layering
        numeric = {}
        for n in nodes:
            v = self._node_attr_safe(self.nx_graph, n, layer_attr)
            try:
                numeric[n] = float(v) if v is not None else None
            except Exception:
                numeric[n] = None

        # Group nodes by integer level; missing -> -1
        keys = {}
        for n, val in numeric.items():
            level = int(val) if (val is not None and not np.isnan(val)) else -1
            keys.setdefault(level, []).append(n)

        # Sort layers descending (highest at top)
        layer_keys = sorted(keys.keys(), reverse=True)

        # Max index for y-spacing
        finite_keys = [k for k in layer_keys if k >= 0]
        max_index = max(finite_keys) if finite_keys else 0

        # Calculate dynamic figure width based on largest layer
        max_nodes_per_layer = max(len(group) for group in keys.values()) if keys else 1
        # Estimate width needed: more nodes need wider figure
        estimated_width = max(figsize[0], min(20, max_nodes_per_layer * 0.8))
        dynamic_figsize = (estimated_width, max(figsize[1], 1.2 * len(layer_keys)))

        # Assign positions: x evenly spaced in layer, y = max_index - level
        pos = {}
        for k in layer_keys:
            group = keys[k]
            m = len(group)
            if m == 1:
                xs = [0.5]
            else:
                # Equally space all nodes in the layer from 0 to 1
                xs = np.linspace(0.0, 1.0, m)

            for i, n in enumerate(group):
                pos[n] = np.array([float(xs[i]), float(k)])

        # Skip parent centering and normalization to maintain equal spacing

        # Prepare coloring
        vals = []
        for n in nodes:
            v = self._node_attr_safe(self.nx_graph, n, color_attr)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        vals = np.array(vals, dtype=float)
        finite = np.isfinite(vals)
        if np.any(finite):
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)
            original_colors = cmap_obj(np.linspace(0, 1, cmap_obj.N))
            lighter_colors = original_colors * 0.5 + 0.5
            cmap_obj = ListedColormap(lighter_colors)
            node_colors = [
                cmap_obj(norm(v)) if np.isfinite(v) else (0.85, 0.85, 0.85, 1.0)
                for v in vals
            ]
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            try:
                sm.set_array([])
            except Exception:
                pass
        else:
            node_colors = "lightgrey"
            sm = None

        # Prepare labels
        labels = {}
        for i, n in enumerate(nodes):
            lv = numeric.get(n)
            base_label = ""
            if lv is not None and not np.isnan(lv):
                if float(lv).is_integer():
                    base_label = str(int(lv))
                else:
                    base_label = f"{lv:.2f}"
            else:
                sid = str(n)
                base_label = sid[:8]
            base_label = f"{i}-" + base_label + format(label_function(self.nx_graph.nodes()[n]['data']))

            # Add state information if requested
            if state_label:
                state_val = self._node_attr_safe(self.nx_graph, n, "state")
                if state_val is not None:
                    try:
                        # Format state as a compact string
                        if hasattr(state_val, "shape") and len(state_val.shape) > 0:
                            # For numpy arrays, show first few elements
                            state_str = np.array2string(
                                state_val,
                                precision=2,
                                separator=",",
                                threshold=3,
                                edgeitems=1,
                            )
                        else:
                            state_str = str(state_val)
                        base_label += f"\n{state_str}"
                    except Exception:
                        base_label += f"\n{str(state_val)[:20]}..."

            labels[n] = base_label

        # Calculate dynamic figure width based on largest layer
        max_nodes_per_layer = max(len(group) for group in keys.values()) if keys else 1
        # Estimate width needed: more nodes need wider figure
        estimated_width = max(figsize[0], min(20, max_nodes_per_layer * 0.8))
        dynamic_figsize = (estimated_width, max(figsize[1], 1.2 * len(layer_keys)))

        # Plot
        fig, ax = plt.subplots(figsize=dynamic_figsize)
        nx.draw_networkx_edges(self.nx_graph, pos, ax=ax, alpha=0.4)
        nx.draw_networkx_nodes(
            self.nx_graph, pos, node_color=node_colors, node_size=200, ax=ax
        )
        if show_labels and len(nodes) <= max_labels:
            nx.draw_networkx_labels(
                self.nx_graph, pos, labels=labels, font_size=8, ax=ax
            )

        if sm is not None:
            try:
                fig.colorbar(sm, ax=ax, label=color_attr)
            except Exception:
                plt.colorbar(sm)

        # Adjust axis limits
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        ax.set_xlim(min(xs) - 0.1, max(xs) + 0.1)
        ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
        ax.axis("off")
        ax.set_title(
            f"Top-down hierarchy by '{layer_attr}' (top = largest {layer_attr})"
        )
        plt.show()

    def visualize_critical_points_by_index(self, indices=None):
        """
        Visualize critical points grouped by their index using py3Dmol.

        Displays 3D molecular structures for each critical point, grouped by index.
        If indices is None, visualizes all indices found in the graph.
        If indices is a list, visualizes only those indices.

        Args:
            indices: List of indices to visualize, or None for all.
        """
        # Collect critical points by index
        index_cps = {}
        for node, attrs in self.nx_graph.nodes(data=True):
            idx = self._node_attr_safe(self.nx_graph, node, "index")
            if idx is not None:
                try:
                    idx = int(idx)
                    if indices is None or idx in indices:
                        index_cps.setdefault(idx, []).append(attrs.get("data", attrs))
                except (ValueError, TypeError):
                    pass

        # Sort indices
        sorted_indices = sorted(index_cps.keys())

        for idx in sorted_indices:
            cps = index_cps[idx]
            name = f"index_{idx}_cp"
            print(f"## {name}")
            for i, cp in enumerate(cps):
                label = ""
                print(f"{name}_{i}, E = {cp.energy:.6f}")
                view = visualize_xyz_py3dmol(cp.xyz, label=label)
                view.show()
