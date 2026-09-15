from __future__ import annotations
from typing import List, Dict
import logging

from .graph import SearchableCriticalPointGraph

# Set up logging
logger = logging.getLogger(__name__)


class NodeStateManager:
    def __init__(self, graph: SearchableCriticalPointGraph):
        self.graph = graph

    def get_unprocessed_nodes(self, procedure: str) -> List[str]:
        return list(self.graph.get_unprocessed_nodes(procedure))

    def get_processed_nodes(self, procedure: str) -> List[str]:
        processed = []
        for node_id in self.graph._graph.nodes:
            if self.graph.is_processed(node_id, procedure):
                processed.append(node_id)
        return processed

    def mark_processing(self, node_id: str, procedure: str) -> None:
        self.graph.mark_processing(node_id, procedure)
        logger.debug(f"Marked node {node_id} as processing for {procedure}")

    def get_processing_nodes(self, procedure: str) -> List[str]:
        return list(self.graph.get_processing_nodes(procedure))

    def is_processing(self, node_id: str, procedure: str) -> bool:
        return self.graph.is_processing(node_id, procedure)

    def mark_processed(self, node_id: str, procedure: str) -> None:
        self.graph.mark_processed(node_id, procedure)
        logger.debug(f"Marked node {node_id} as processed for {procedure}")

    def is_processed(self, node_id: str, procedure: str) -> bool:
        return self.graph.is_processed(node_id, procedure)

    def get_processing_stats(self, procedure: str) -> Dict[str, int]:
        return self.graph.get_processing_stats(procedure)

    def has_unprocessed_nodes(self, procedure: str) -> bool:
        return len(self.get_unprocessed_nodes(procedure)) > 0
