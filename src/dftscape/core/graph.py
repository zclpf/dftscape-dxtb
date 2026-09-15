import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import json
import bisect

from dftscape.core.point import CriticalPoint
from dftscape.core.interfaces import Backend
from dftscape.common.distance import DistanceFunction, DistanceRegistry

logger = logging.getLogger(__name__)


class CriticalPointGraph:

    def __init__(
        self,
        threshold: float = 0.1,
        distance_function: str | DistanceFunction = "rmsd",
        energy_threshold: Optional[float] = 0.005,  # e.g. 0.005 Hartree
        **distance_kwargs: Any,
    ):
        self._graph = nx.DiGraph()
        self.threshold = threshold
        self.energy_threshold = energy_threshold
        # Sorted list of (energy, node_id) for fast candidate lookup
        self._energy_index: list[tuple[float, str]] = []

        # Configure distance function
        if isinstance(distance_function, str):
            self.distance_function = DistanceRegistry.create(distance_function)
            self.distance_function_name = distance_function
        else:
            self.distance_function = distance_function
            self.distance_function_name = getattr(
                distance_function, "__name__", "custom"
            )

        self.distance_kwargs = distance_kwargs

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX graph."""
        return self._graph

    def add(self, critical_point: CriticalPoint) -> bool:
        new_xyz = critical_point.xyz

        logger.debug("Attempting to add critical point:")
        logger.debug(f"  Index: {critical_point.index}")
        logger.debug(f"  Energy: {critical_point.energy:.6f}")
        logger.debug(f"  Position: {critical_point.state}")
        logger.debug(f"  XYZ key: {new_xyz}")
        logger.debug(f"  Distance function: {self.distance_function_name}")
        logger.debug(f"  Threshold: {self.threshold}")

        # Check if this exact XYZ already exists
        if new_xyz in self._graph.nodes:
            existing_point = self._graph.nodes[new_xyz]["data"]
            logger.debug("Exact XYZ match found - point already exists!")
            logger.debug(
                f"  Existing point: index={existing_point.index}, energy={existing_point.energy:.6f}"
            )
            logger.debug(
                f"Critical point with identical XYZ already exists (index: {existing_point.index})"
            )
            return False

        logger.debug(
            f"No exact XYZ match. Checking similarity with {len(self._graph.nodes)} existing nodes:"
        )

        # Determine which nodes to compare against
        candidates = self._get_candidates(critical_point)

        for existing_xyz in candidates:
            existing_point = self._graph.nodes[existing_xyz]["data"]
            logger.debug("  Comparing with existing node:")
            logger.debug(f"    Existing XYZ: {existing_xyz}")
            logger.debug(
                f"    Existing point: index={existing_point.index}, energy={existing_point.energy:.6f}, position={existing_point.state}"
            )

            distance = self.distance_function(
                critical_point, existing_point, **self.distance_kwargs # type: ignore
            )
            logger.debug(f"    Distance computed: {distance:.8f}")
            logger.debug(f"    Threshold: {self.threshold}")
            logger.debug(f"    Distance <= threshold: {distance <= self.threshold}")

            # Debug: Parse coordinates to verify
            #from ..common.distance import get_coordinates_xyz_lines

            #lines1 = new_xyz.strip().split("\n")
            #lines2 = existing_xyz.strip().split("\n")
            #atoms1, coords1 = get_coordinates_xyz_lines(lines1)
            #atoms2, coords2 = get_coordinates_xyz_lines(lines2)
            #logger.debug(f"    New coords: {coords1}")
            #logger.debug(f"    Existing coords: {coords2}")
            #logger.debug(
            #    f"    Manual distance: {np.linalg.norm(coords1 - coords2):.8f}"
            #)

            if distance <= self.threshold:
                logger.debug("DEBUG: Point is too similar - NOT adding to graph!")
                logger.debug(
                    f"  Reason: distance {distance:.8f} <= threshold {self.threshold}"
                )
                logger.debug(
                    f"Critical point too similar to existing node (distance: {distance:.4f}, index: {existing_point.index})"
                )
                return False

        # If we get here, the point is unique enough - add it
        logger.debug("DEBUG: Point is unique enough - adding to graph!")
        self._graph.add_node(new_xyz, data=critical_point)
        bisect.insort(self._energy_index, (critical_point.energy, new_xyz))
        logger.debug(
            f"Added unique critical point (index: {critical_point.index}, energy: {critical_point.energy:.4f})"
        )
        return True

    def _get_candidates(self, critical_point: CriticalPoint) -> list[str]:
        """
        Return node IDs that are worth running the distance function against.
        Filters candidates based on energy threshold (if set) and ensures they
        share either the same index or the same number of fragments.
        """
        target_index = critical_point.index
        target_fragments = critical_point.num_fragments

        # Apply energy filter
        e = critical_point.energy
        lo = bisect.bisect_left(self._energy_index,  (e - self.energy_threshold, ""))
        hi = bisect.bisect_right(self._energy_index, (e + self.energy_threshold, "\xff"))
        candidate_ids = [node_id for _, node_id in self._energy_index[lo:hi]]

        # Filter by index OR num_fragments
        filtered_candidates = []
        if target_fragments == 1:
            for node_id in candidate_ids:
                node_data = self._graph.nodes[node_id]["data"]                
                if node_data.index == target_index:
                    filtered_candidates.append(node_id)
        else:
            for node_id in candidate_ids:
                node_data = self._graph.nodes[node_id]["data"]                
                if node_data.num_fragments == target_fragments:
                    filtered_candidates.append(node_id)
            
        return filtered_candidates

    def find_similar(
        self, target_point: CriticalPoint, threshold: float = None
    ) -> List[Tuple[str, float, CriticalPoint]]:
        if threshold is None:
            threshold = self.threshold

        target_xyz = target_point.xyz
        similar = []

        for node_xyz in self._graph.nodes:
            if node_xyz == target_xyz:
                continue  # Skip self-comparison

            distance = self.distance_function(
                target_point, self._graph.nodes[node_xyz]["data"], **self.distance_kwargs # type: ignore
            )
            if distance <= threshold:
                similar.append(
                    (node_xyz, distance, self._graph.nodes[node_xyz]["data"])
                )

        return sorted(similar, key=lambda x: x[1])  # Sort by distance

    def get_critical_points_by_index(self) -> Dict[int, List[CriticalPoint]]:
        points_by_index = {}

        for node_data in self._graph.nodes.values():
            point = node_data["data"]
            index = point.index

            if index not in points_by_index:
                points_by_index[index] = []

            points_by_index[index].append(point)

        # Sort points within each index by energy
        for index in points_by_index:
            points_by_index[index].sort(key=lambda p: p.energy)

        return points_by_index

    def check_uniqueness(self, threshold: float = None) -> Dict[str, Any]:
        if threshold is None:
            threshold = self.threshold

        nodes_list = list(self._graph.nodes)
        duplicates = []

        logger.debug(f"Checking uniqueness of {len(nodes_list)} nodes...")

        for i, xyz1 in enumerate(nodes_list):
            for j, xyz2 in enumerate(nodes_list[i + 1 :]):
                distance = self.distance_function(self._graph.nodes[xyz1]["data"], 
                                                  self._graph.nodes[xyz2]["data"], **self.distance_kwargs) # type: ignore
                if distance <= threshold:
                    point1 = self._graph.nodes[xyz1]["data"]
                    point2 = self._graph.nodes[xyz2]["data"]
                    duplicates.append(
                        {
                            "distance": distance,
                            "point1": {"index": point1.index, "energy": point1.energy, "xyz": xyz1, "i": i},
                            "point2": {"index": point2.index, "energy": point2.energy, "xyz": xyz2, "j": i+1+j},
                        }
                    )
                    logger.debug(
                        f"Duplicate found: distance {distance:.4f} between indices {point1.index} and {point2.index}"
                    )

        result = {
            "total_nodes": len(nodes_list),
            "duplicates_found": len(duplicates),
            "duplicates": duplicates,
        }

        logger.debug("\nSummary:")
        logger.debug(f"- Total nodes: {result['total_nodes']}")
        logger.debug(f"- Duplicate pairs: {result['duplicates_found']}")

        return result

    def save(self, filepath: str) -> None:
        logger = logging.getLogger(__name__)

        graph_data = self.to_config()

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(graph_data, f, indent=2)

        logger.debug(f"Saved CriticalPointGraph to {filepath} (JSON format)")

    def to_config(self) -> Dict[str, Any]:

        # Prepare graph data
        graph_data = {
            "threshold": self.threshold,
            "distance_function": self.distance_function_name,
            "distance_kwargs": self.distance_kwargs,
            "nodes": {},
            "edges": [],
            "energy_threshold" : self.energy_threshold
        }

        # Serialize nodes with CriticalPoint data
        for node_id, node_data in self._graph.nodes(data=True):
            critical_point = node_data["data"]
            graph_data["nodes"][node_id] = critical_point.to_config()

        # Serialize edges
        for u, v, edge_data in self._graph.edges(data=True):
            # Process edge data to handle SerialisableDataClass objects
            processed_edge_data = {}
            for key, value in edge_data.items():
                if hasattr(value, "to_config"):  # Check if it's a SerialisableDataClass
                    processed_edge_data[key] = value.to_config()
                else:
                    processed_edge_data[key] = value

            graph_data["edges"].append(
                {"source": u, "target": v, "data": processed_edge_data}
            )

        return graph_data

    @classmethod
    def from_config(cls, graph_data: Dict[str, Any]) -> "CriticalPointGraph":

        # Create new graph instance
        graph = cls(
            threshold=graph_data["threshold"],
            distance_function=graph_data.get("distance_function", "rmsd"),
            energy_threshold=graph_data.get("energy_threshold"),
            **graph_data.get("distance_kwargs", {}),
        )

        # Deserialize nodes
        for node_id, node_config in graph_data["nodes"].items():
            critical_point = CriticalPoint.from_config(node_config)
            graph._graph.add_node(node_id, data=critical_point)
            bisect.insort(graph._energy_index, (critical_point.energy, node_id))

        # Deserialize edges
        for edge_data in graph_data["edges"]:
            # Process edge data to reconstruct SerialisableDataClass objects
            processed_edge_data = {}
            for key, value in edge_data["data"].items():
                if isinstance(value, dict) and "_class" in value:
                    # Import the class dynamically
                    module_name = value["_module"]
                    class_name = value["_class"]
                    module = __import__(module_name, fromlist=[class_name])
                    item_class = getattr(module, class_name)
                    processed_edge_data[key] = item_class.from_config(value)
                else:
                    processed_edge_data[key] = value

            graph._graph.add_edge(
                edge_data["source"], edge_data["target"], **processed_edge_data
            )

        return graph

    # Delegate other NetworkX methods to the underlying graph
    def __getattr__(self, name):
        return getattr(self._graph, name)

    def __len__(self):
        return len(self._graph)

    def __iter__(self):
        return iter(self._graph)

    def __contains__(self, item):
        return item in self._graph

    def _format_xyz_short(self, xyz_str: str, max_atoms: int = 3) -> str:
        lines = xyz_str.strip().split("\n")
        if len(lines) < 2:
            return xyz_str

        try:
            num_atoms = int(lines[0])
            comment = lines[1] if len(lines) > 1 else ""
            atom_lines = lines[2:]

            if num_atoms <= max_atoms:
                # Show all atoms
                formatted_atoms = []
                for line in atom_lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_type = parts[0]
                        x, y, z = parts[1:4]
                        formatted_atoms.append(f"{atom_type}({x[:4]},{y[:4]},{z[:4]})")
                return f"{num_atoms} atoms: {' '.join(formatted_atoms)}"
            else:
                # Show first few atoms and indicate truncation
                formatted_atoms = []
                for line in atom_lines[:max_atoms]:
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_type = parts[0]
                        x, y, z = parts[1:4]
                        formatted_atoms.append(f"{atom_type}({x[:4]},{y[:4]},{z[:4]})")
                return f"{num_atoms} atoms: {' '.join(formatted_atoms)} ... (+{num_atoms-max_atoms} more)"

        except (ValueError, IndexError):
            # Fallback if parsing fails
            return xyz_str[:50] + "..." if len(xyz_str) > 50 else xyz_str

    def report(self) -> str:
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Critical Point Graph Summary")
        lines.append("=" * 80)

        # Basic graph statistics
        total_nodes = len(self._graph.nodes)
        total_edges = len(self._graph.edges)
        lines.append("\nGraph Structure:")
        lines.append(f"  Total nodes: {total_nodes}")
        lines.append(f"  Total edges: {total_edges}")
        lines.append(f"  Distance function: {self.distance_function_name}")
        lines.append(f"  Distance threshold: {self.threshold}")
        if self.distance_kwargs:
            lines.append(f"  Distance kwargs: {self.distance_kwargs}")

        # Node analysis
        if total_nodes > 0:
            intact_index_counts = {}
            fragmented_points = []
            energy_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0}

            for node_id, node_data in self._graph.nodes(data=True):
                point = node_data["data"]
                index = point.index
                energy = point.energy
                is_fragmented = getattr(point, "num_fragments", 1) > 1

                # Separate the intact molecules from the fragmented endpoints
                if is_fragmented:
                    fragmented_points.append((point, energy))
                else:
                    if index not in intact_index_counts:
                        intact_index_counts[index] = []
                    intact_index_counts[index].append((point, energy))

                # Energy statistics
                energy_stats["min"] = min(energy_stats["min"], energy)
                energy_stats["max"] = max(energy_stats["max"], energy)
                energy_stats["sum"] += energy

            # --- Print Intact Molecules ---
            if intact_index_counts:
                lines.append("\nIntact Molecules by Morse Index:")
                for index in sorted(intact_index_counts.keys()):
                    points = intact_index_counts[index]
                    count = len(points)
                    energies = [e for _, e in points]
                    min_energy = min(energies)
                    max_energy = max(energies)
                    lines.append(
                        f"  Index {index}: {count} points - Energy range: [{min_energy:.4f}, {max_energy:.4f}]"
                    )

                    display_limit = 3 if count <= 3 else 2
                    for point, energy in points[:display_limit]:
                        xyz_short = self._format_xyz_short(point.xyz)
                        lines.append(f"    {xyz_short} (E={energy:.4f})")
                        
                    if count > display_limit:
                        lines.append(f"    ... (+{count - display_limit} more)")

            # --- Print Fragmented Products ---
            if fragmented_points:
                count = len(fragmented_points)
                energies = [e for _, e in fragmented_points]
                min_energy = min(energies)
                max_energy = max(energies)
                lines.append("\nFragmented Products (Reaction Endpoints):")
                lines.append(
                    f"  Total: {count} points - Energy range: [{min_energy:.4f}, {max_energy:.4f}]"
                )
                
                display_limit = 3 if count <= 3 else 2
                for point, energy in fragmented_points[:display_limit]:
                    xyz_short = self._format_xyz_short(point.xyz)
                    lines.append(f"    {xyz_short} ({point.num_fragments} frags, E={energy:.4f})")
                    
                if count > display_limit:
                    lines.append(f"    ... (+{count - display_limit} more)")

            lines.append("\nEnergy Statistics (All Nodes):")
            lines.append(
                f"  Energy range: [{energy_stats['min']:.4f}, {energy_stats['max']:.4f}]"
            )
            lines.append(
                f"  Energy span: {energy_stats['max'] - energy_stats['min']:.4f}"
            )
            lines.append(f"  Average energy: {energy_stats['sum'] / total_nodes:.4f}")

        # Edge connectivity analysis
        if total_edges > 0:
            lines.append("\nConnectivity Analysis:")
            degrees = [deg for _, deg in self._graph.degree()]
            if degrees:
                lines.append(f"  Average degree: {sum(degrees) / len(degrees):.2f}")
                lines.append(f"  Max degree: {max(degrees)}")
                lines.append(f"  Min degree: {min(degrees)}")

            # Edge weight statistics
            edge_weights = []
            for _, _, edge_data in self._graph.edges(data=True):
                if "distance" in edge_data:
                    edge_weights.append(edge_data["distance"])
                elif "weight" in edge_data:
                    edge_weights.append(edge_data["weight"])

            if edge_weights:
                lines.append(
                    f"  Edge weight range: [{min(edge_weights):.4f}, {max(edge_weights):.4f}]"
                )
                lines.append(
                    f"  Average edge weight: {sum(edge_weights) / len(edge_weights):.4f}"
                )

        lines.append("=" * 80)
        return "\n".join(lines)


class SearchableCriticalPointGraph(CriticalPointGraph):
    def __init__(
        self,
        threshold: float = 0.1,
        distance_function: str | DistanceFunction = "rmsd",
        **distance_kwargs: Any,
    ):
        super().__init__(threshold, distance_function, **distance_kwargs)
        # Dictionary of search procedures, each with their own state
        self._search_states: Dict[str, Dict[str, Set[str]]] = {}

    def _get_search_state(self, search_procedure: str) -> Dict[str, Set[str]]:
        if search_procedure not in self._search_states:
            self._search_states[search_procedure] = {
                "processed_nodes": set(),
                "processing_nodes": set(),
            }
        return self._search_states[search_procedure]

    def mark_processed(self, node_id: str, search_procedure: str) -> None:
        if node_id in self._graph.nodes:
            state = self._get_search_state(search_procedure)
            state["processed_nodes"].add(node_id)
            # Remove from processing if it was there
            state["processing_nodes"].discard(node_id)
        else:
            logger.warning(
                f"mark_processed({search_procedure}): node {node_id[:50]}... not in graph, not marking"
            )

    def mark_processing(self, node_id: str, search_procedure: str) -> None:
        if node_id in self._graph.nodes:
            state = self._get_search_state(search_procedure)
            state["processing_nodes"].add(node_id)
        else:
            logger.warning(
                f"mark_processing({search_procedure}): node {node_id[:50]}... not in graph, not marking"
            )

    def is_processed(self, node_id: str, search_procedure: str) -> bool:
        state = self._get_search_state(search_procedure)
        return node_id in state["processed_nodes"]

    def is_processing(self, node_id: str, search_procedure: str) -> bool:
        state = self._get_search_state(search_procedure)
        return node_id in state["processing_nodes"]

    def get_unprocessed_nodes(self, search_procedure: str) -> Set[str]:
        state = self._get_search_state(search_procedure)
        all_nodes = set(self._graph.nodes)
        processed = state["processed_nodes"]
        processing = state["processing_nodes"]
        unprocessed = all_nodes - processed - processing
        return unprocessed

    def get_processing_nodes(self, search_procedure: str) -> Set[str]:
        state = self._get_search_state(search_procedure)
        return state["processing_nodes"].copy()

    def get_next_to_process(self, search_procedure: str) -> Optional[str]:
        unprocessed = self.get_unprocessed_nodes(search_procedure)
        return next(iter(unprocessed)) if unprocessed else None

    def get_search_procedures(self) -> List[str]:
        return list(self._search_states.keys())

    def get_processing_stats(self, search_procedure: str) -> Dict[str, int]:
        state = self._get_search_state(search_procedure)
        total_nodes = len(self._graph.nodes)
        processed = len(state["processed_nodes"])
        processing = len(state["processing_nodes"])
        unprocessed = total_nodes - processed - processing

        return {
            "total_nodes": total_nodes,
            "processed": processed,
            "processing": processing,
            "unprocessed": unprocessed,
        }  # def _save_json(self, filepath: str) -> None:

    def to_config(self) -> Dict[str, Any]:
        graph_data = super().to_config()

        # Add search states
        graph_data["search_states"] = {}
        for procedure, state in self._search_states.items():
            graph_data["search_states"][procedure] = {
                "processed_nodes": list(state["processed_nodes"]),
                "processing_nodes": list(state["processing_nodes"]),
            }

        return graph_data

    @classmethod
    def from_config(cls, graph_data) -> "SearchableCriticalPointGraph":

        # Create instance using parent method
        instance = super(SearchableCriticalPointGraph, cls).from_config(graph_data)

        # Restore search states if present
        if "search_states" in graph_data:
            instance._search_states = {}
            for procedure, state_data in graph_data["search_states"].items():
                instance._search_states[procedure] = {
                    "processed_nodes": set(state_data["processed_nodes"]),
                    "processing_nodes": set(state_data.get("processing_nodes", [])),
                }
        else:
            # Initialize empty search states if none saved
            instance._search_states = {}

        return instance

    def report(self, show_search_status: bool = False) -> str:
        # Get the base report
        base_report = super().report()

        # Add search-specific information
        lines = [base_report]

        if show_search_status:
            # Search procedure statistics
            search_procedures = self.get_search_procedures()
            if search_procedures:
                lines.append("\nSearch Procedure Status:")
                for procedure in sorted(search_procedures):
                    stats = self.get_processing_stats(procedure)
                    lines.append(f"  {procedure.title()} search:")
                    lines.append(
                        f"    Processed: {stats['processed']}, Processing: {stats['processing']}, Unprocessed: {stats['unprocessed']}"
                    )
                    lines.append(
                        f"    Completion: {(stats['processed'] / stats['total_nodes'] * 100):.1f}%"
                    )

                    # Show unprocessed nodes if any
                    if stats["unprocessed"] > 0:
                        unprocessed_nodes = list(self.get_unprocessed_nodes(procedure))[
                            :3
                        ]
                        if len(unprocessed_nodes) > 0:
                            lines.append("    Next to process:")
                            for node_id in unprocessed_nodes:
                                point = self._graph.nodes[node_id]["data"]
                                coords = [round(float(c), 1) for c in point.state]
                                lines.append(
                                    f"      Index {point.index}: [{coords[0]} {coords[1]} {coords[2]}] (E={point.energy:.4f})"
                                )
                            if stats["unprocessed"] > 3:
                                lines.append(
                                    f"      ... (+{stats['unprocessed'] - 3} more)"
                                )
            else:
                lines.append("\nSearch Procedure Status:")
                lines.append("  No search procedures registered yet")

        return "\n".join(lines)
