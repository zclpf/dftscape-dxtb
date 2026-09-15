from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class OrchestratorCallback(ABC):
    @abstractmethod
    def on_state_change(self, orchestrator) -> None:
        pass


class SearchCheckpointerCallback(OrchestratorCallback):

    def __init__(
        self,
        output_dir: str,
        enabled: bool = False,
        interval: int = 10,
        max_count: int = 10,
        prefix: str = "checkpoint_periodic",
    ):
        self.enabled = enabled
        self.interval = interval
        self.max_count = max_count
        self.prefix = prefix
        self.output_dir = Path(output_dir)

        # Internal state for tracking checkpoints
        self.last_checkpoint_job = 0

    def on_state_change(self, orchestrator) -> None:
        if not self.enabled:
            return

        # Get current total jobs from monitor
        upward_stats = orchestrator.monitor.search_stats.get("upward")
        downward_stats = orchestrator.monitor.search_stats.get("downward")
        upward_jobs = upward_stats.total_jobs if upward_stats else 0
        downward_jobs = downward_stats.total_jobs if downward_stats else 0
        current_job_count = upward_jobs + downward_jobs

        # Check if we should save a checkpoint (every interval jobs from start)
        next_checkpoint_job = (
            (self.last_checkpoint_job // self.interval) + 1
        ) * self.interval
        if current_job_count >= next_checkpoint_job:
            # Periodic checkpoint
            self.save_checkpoint(orchestrator, current_job_count, is_periodic=True)
            self.last_checkpoint_job = next_checkpoint_job
            self.cleanup_old_checkpoints()

    def save_checkpoint(
        self,
        orchestrator,
        job_count: int,
        is_periodic: bool = True,
        name: str | None = None,
        suffix: str | None = None,
    ) -> None:

        def _make_periodic_name(count: int) -> str:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.prefix}_{count:06d}_{ts}"

        if name is not None:
            checkpoint_dir_name = name
        else:
            if is_periodic:
                checkpoint_dir_name = _make_periodic_name(job_count)
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = f"checkpoint_manual_{job_count:06d}_{ts}"
                checkpoint_dir_name = f"{base}_{suffix}" if suffix else base

        checkpoint_dir = self.output_dir / checkpoint_dir_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save graph to graph.json
        graph_path = checkpoint_dir / "graph.json"
        with open(graph_path, "w") as f:
            json.dump(orchestrator.graph.to_config(), f, indent=2)

        # Save job_queues to job_queues.json
        job_queues_path = checkpoint_dir / "job_queues.json"
        with open(job_queues_path, "w") as f:
            json.dump(orchestrator.job_queues.to_config(), f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def cleanup_old_checkpoints(self) -> None:
        if self.max_count <= 0:
            return
        # Only consider periodic checkpoints matching the configured prefix
        checkpoint_pattern = f"{self.prefix}_*"
        checkpoint_dirs = [
            d for d in self.output_dir.glob(checkpoint_pattern) if d.is_dir()
        ]

        if len(checkpoint_dirs) > self.max_count:
            # Sort by modification time, keep the most recent max_count
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            dirs_to_remove = checkpoint_dirs[self.max_count :]

            for old_dir in dirs_to_remove:
                # Use shutil to remove directory tree
                import shutil

                shutil.rmtree(old_dir)
                logger.info(f"Removed old periodic checkpoint directory: {old_dir}")

    def trigger_manual_checkpoint(
        self, orchestrator, job_procedure: str = None, suffix: str | None = None
    ) -> None:
        if not self.enabled:
            return

        # Compute current job count and save a manual checkpoint that will
        # not be subject to periodic cleanup.
        upward_stats = orchestrator.monitor.search_stats.get("upward")
        downward_stats = orchestrator.monitor.search_stats.get("downward")
        upward_jobs = upward_stats.total_jobs if upward_stats else 0
        downward_jobs = downward_stats.total_jobs if downward_stats else 0
        current_job_count = upward_jobs + downward_jobs

        # Save manual checkpoint under a distinct prefix so it won't be cleaned
        # up by the periodic cleanup logic. Delegate to `save_checkpoint`.
        try:
            self.save_checkpoint(
                orchestrator, current_job_count, is_periodic=False, suffix=suffix
            )
        except Exception as e:
            logger.warning(f"Failed to save manual checkpoint: {e}")

    def get_checkpoint_summary(self) -> tuple[int, list[str]]:
        if not self.enabled:
            return 0, []

        checkpoint_pattern = f"{self.prefix}_*"
        checkpoint_dirs = [
            d for d in self.output_dir.glob(checkpoint_pattern) if d.is_dir()
        ]

        return len(checkpoint_dirs), [
            d.name
            for d in sorted(
                checkpoint_dirs, key=lambda x: x.stat().st_mtime, reverse=True
            )
        ]

    @staticmethod
    def load_checkpoint(checkpoint_dir: str | Path) -> tuple:
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_path}"
            )

        graph_file = checkpoint_path / "graph.json"
        job_queues_file = checkpoint_path / "job_queues.json"

        if not graph_file.exists():
            raise FileNotFoundError(
                f"graph.json not found in checkpoint directory: {checkpoint_path}"
            )

        if not job_queues_file.exists():
            raise FileNotFoundError(
                f"job_queues.json not found in checkpoint directory: {checkpoint_path}"
            )

        # Import here to avoid circular imports
        from .graph import SearchableCriticalPointGraph
        from .job_queues import JobQueues

        try:
            # Load and parse graph.json
            with open(graph_file, "r") as f:
                graph_config = json.load(f)

            # Load and parse job_queues.json
            with open(job_queues_file, "r") as f:
                job_queues_config = json.load(f)

            # Reconstruct objects from config
            graph = SearchableCriticalPointGraph.from_config(graph_config)
            job_queues = JobQueues.from_config(job_queues_config)

            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            return graph, job_queues

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in checkpoint files: {e}")


class SearchInfoCallback(OrchestratorCallback):

    def on_state_change(self, orchestrator) -> None:
        logger.info("=== Search Status Update ===")
        logger.info(orchestrator.report())
        logger.info("=== End Search Status Update ===\n")
