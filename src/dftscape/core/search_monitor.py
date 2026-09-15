from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
import logging

if TYPE_CHECKING:
    from .search_orchestrator import SearchOrchestrator

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SearchStatistics:

    procedure: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_execution_time: float = 0.0
    average_job_time: float = 0.0
    min_job_time: float = float("inf")
    max_job_time: float = 0.0

    @property
    def duration(self) -> Optional[float]:
        """Get the total duration of the search in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if self.total_jobs == 0:
            return 0.0
        return ((self.total_jobs - self.failed_jobs) / self.total_jobs) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if the search is complete."""
        return self.end_time is not None


@dataclass
class ProgressReport:

    procedure: str
    timestamp: datetime
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    running_jobs: int = 0
    queued_jobs: int = 0
    total_execution_time: float = 0.0
    average_job_time: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    current_memory_usage: Optional[float] = None
    current_cpu_usage: Optional[float] = None
    # Additional fields for detailed progress reporting
    total_nodes: int = 0
    processed_nodes: int = 0
    processing_nodes: int = 0
    unprocessed_nodes: int = 0
    pending_jobs: int = 0
    elapsed_time: float = 0.0
    estimated_time_remaining: Optional[float] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate the percentage of nodes processed or processing."""
        if self.total_nodes == 0:
            return 100.0
        return (
            (self.processed_nodes + self.processing_nodes) / self.total_nodes
        ) * 100.0

    @property
    def remaining_jobs(self) -> int:
        """Calculate the number of jobs remaining."""
        return self.pending_jobs

    @property
    def is_complete(self) -> bool:
        """Check if all jobs have been completed."""
        return self.pending_jobs == 0


class SearchMonitor:

    def __init__(self):
        self.search_stats: Dict[str, SearchStatistics] = {}
        self.job_history: List[Dict] = []
        self.enabled = True

    def start_search(self, procedure: str) -> None:
        if not self.enabled:
            return

        self.search_stats[procedure] = SearchStatistics(
            procedure=procedure, start_time=datetime.now()
        )
        logger.debug(f"Started monitoring search: {procedure}")

    def end_search(self, procedure: str) -> None:
        if not self.enabled or procedure not in self.search_stats:
            return

        self.search_stats[procedure].end_time = datetime.now()

        # Calculate final statistics
        stats = self.search_stats[procedure]
        if stats.total_jobs > 0:
            stats.average_job_time = stats.total_execution_time / stats.total_jobs

        logger.debug(f"Ended monitoring search: {procedure}")
        logger.debug(f"Search statistics: {stats}")

    def record_job_completion(
        self,
        procedure: str,
        job_id: str,
        success: bool,
        execution_time: float,
        node_id: str,
    ) -> None:
        if not self.enabled or procedure not in self.search_stats:
            return

        stats = self.search_stats[procedure]
        stats.total_jobs += 1
        stats.total_execution_time += execution_time

        if success:
            stats.completed_jobs += 1
        else:
            stats.failed_jobs += 1

        # Update min/max times
        if execution_time < stats.min_job_time:
            stats.min_job_time = execution_time
        if execution_time > stats.max_job_time:
            stats.max_job_time = execution_time

        # Record job history
        self.job_history.append(
            {
                "procedure": procedure,
                "job_id": job_id,
                "node_id": node_id,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now(),
            }
        )

    def get_progress_report(
        self, procedure: str, orchestrator: "SearchOrchestrator"
    ) -> Optional[ProgressReport]:
        if not self.enabled or procedure not in self.search_stats:
            return None

        stats = self.search_stats[procedure]
        current_time = datetime.now()
        elapsed_time = (current_time - stats.start_time).total_seconds()

        # Get node counts from orchestrator
        total_nodes = len(orchestrator.graph._graph.nodes)
        processed_nodes = len(orchestrator.node_states.get_processed_nodes(procedure))
        processing_nodes = len(orchestrator.node_states.get_processing_nodes(procedure))
        unprocessed_nodes = len(
            orchestrator.node_states.get_unprocessed_nodes(procedure)
        )

        # Get job counts
        total_jobs = stats.total_jobs
        completed_jobs = stats.completed_jobs
        failed_jobs = stats.failed_jobs
        pending_jobs = total_jobs - completed_jobs - failed_jobs

        # Estimate time remaining
        estimated_time_remaining = None
        if processed_nodes > 0 and elapsed_time > 0:
            avg_time_per_node = elapsed_time / (processed_nodes + processing_nodes)
            estimated_time_remaining = avg_time_per_node * unprocessed_nodes

        return ProgressReport(
            procedure=procedure,
            timestamp=current_time,
            total_nodes=total_nodes,
            processed_nodes=processed_nodes,
            processing_nodes=processing_nodes,
            unprocessed_nodes=unprocessed_nodes,
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            pending_jobs=pending_jobs,
            failed_jobs=failed_jobs,
            elapsed_time=elapsed_time,
            estimated_time_remaining=estimated_time_remaining,
        )

    def get_search_statistics(self, procedure: str) -> Optional[SearchStatistics]:
        return self.search_stats.get(procedure)

    def get_all_search_statistics(self) -> Dict[str, SearchStatistics]:
        return self.search_stats.copy()

    def get_job_history(self, procedure: Optional[str] = None) -> List[Dict]:
        if procedure is None:
            return self.job_history.copy()
        return [job for job in self.job_history if job["procedure"] == procedure]

    def report(self, orchestrator: Optional["SearchOrchestrator"] = None) -> str:
        if not self.enabled:
            return "Monitoring is disabled"

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Search Monitor Summary")
        lines.append("=" * 80)

        for procedure, stats in self.search_stats.items():
            lines.append(f"Search Procedure: {procedure}")
            lines.append(
                f"  Duration: {stats.duration:.1f}s"
                if stats.duration
                else "  Duration: In progress"
            )
            lines.append(f"  Total Jobs: {stats.total_jobs}")
            lines.append(f"  Completed: {stats.completed_jobs}")
            lines.append(f"  Failed: {stats.failed_jobs}")
            lines.append(f"  Success Rate: {stats.success_rate:.1f}%")

            if stats.total_jobs > 0:
                lines.append(f"  Average Job Time: {stats.average_job_time:.3f}s")
                lines.append(f"  Min Job Time: {stats.min_job_time:.3f}s")
                lines.append(f"  Max Job Time: {stats.max_job_time:.3f}s")
                lines.append(
                    f"  Total Execution Time: {stats.total_execution_time:.3f}s"
                )

            # Add node processing information if orchestrator is provided
            if orchestrator is not None:
                try:
                    processed_nodes = len(orchestrator.node_states.get_processed_nodes(procedure))
                    processing_nodes = len(orchestrator.node_states.get_processing_nodes(procedure))
                    unprocessed_nodes = len(orchestrator.node_states.get_unprocessed_nodes(procedure))
                    total_nodes = len(orchestrator.graph._graph.nodes)

                    lines.append(f"  Node Status: {processed_nodes} processed, {processing_nodes} processing, {unprocessed_nodes} unprocessed (total: {total_nodes})")
                    if total_nodes > 0:
                        progress_pct = ((processed_nodes + processing_nodes) / total_nodes) * 100
                        lines.append(f"  Progress: {progress_pct:.1f}%")
                except Exception as e:
                    lines.append(f"  Node status unavailable: {e}")

        if not self.search_stats:
            lines.append("No search operations monitored yet.")

        lines.append("=" * 80)

        return "\n".join(lines)

    def reset(self) -> None:
        self.search_stats.clear()
        self.job_history.clear()
        logger.debug("Search monitor reset")

    def enable(self) -> None:
        self.enabled = True
        logger.debug("Search monitoring enabled")

    def disable(self) -> None:
        self.enabled = False
        logger.debug("Search monitoring disabled")
