from __future__ import annotations
from typing import List, Dict, Set, Optional, TYPE_CHECKING
import logging

from .job_result import JobResult

if TYPE_CHECKING:
    from .job_queues import JobQueues

# Set up logging
logger = logging.getLogger(__name__)


class JobTracker:

    def __init__(self):
        # node_id -> set of job_ids that were spawned from this node
        self.pending_jobs: Dict[str, Set[str]] = {}
        # node_id -> set of job_ids that have completed
        self.completed_jobs: Dict[str, Set[str]] = {}
        # job_id -> JobResult for completed jobs
        self.job_results: Dict[str, JobResult] = {}

    def add_jobs_for_node(self, node_id: str, job_ids: List[str]) -> None:
        if node_id not in self.pending_jobs:
            self.pending_jobs[node_id] = set()
        if node_id not in self.completed_jobs:
            self.completed_jobs[node_id] = set()

        self.pending_jobs[node_id].update(job_ids)
        logger.debug(f"Added {len(job_ids)} jobs for node {node_id}")

    def mark_job_complete(self, node_id: str, job_id: str, result: JobResult) -> None:
        # Normalize job_id to string
        job_id_str = str(job_id)

        # If already completed, skip
        if job_id_str in self.job_results:
            logger.debug(f"Job {job_id_str} already completed, skipping")
            return

        # Direct hit
        if node_id in self.pending_jobs and job_id_str in self.pending_jobs[node_id]:
            self.pending_jobs[node_id].remove(job_id_str)
            self.completed_jobs[node_id].add(job_id_str)
            self.job_results[job_id_str] = result
            logger.debug(f"✓ Job {job_id_str} completed for node {node_id}")
            return

        # Fallback: sometimes callers pass only a numeric suffix (e.g. '1') instead
        # of the full job id 'node_upward_1'. Try to match by suffix within pending.
        pending_for_node = self.pending_jobs.get(node_id, set())
        if pending_for_node:
            # Look for any pending job that ends with the provided id
            candidate = None
            for pj in pending_for_node:
                if pj == job_id_str:
                    candidate = pj
                    break
                # match suffix like '_1' or '_upward_1' -> split last token
                try:
                    last_token = pj.rsplit("_", 1)[-1]
                except Exception:
                    last_token = pj
                if last_token == job_id_str:
                    candidate = pj
                    break

            if candidate is not None:
                # Use the matched candidate id for bookkeeping
                self.pending_jobs[node_id].remove(candidate)
                self.completed_jobs.setdefault(node_id, set()).add(candidate)
                self.job_results[candidate] = result
                logger.debug(
                    f"✓ Job {candidate} (matched from '{job_id_str}') completed for node {node_id}"
                )
                return

        # Nothing matched - provide diagnostic context in the warning
        logger.warning(
            f"✗ Attempted to mark unknown job {job_id_str} as complete for node {node_id}."
            f" Pending jobs for node: {sorted(list(pending_for_node))}"
        )

    def all_jobs_complete(self, node_id: str) -> bool:
        pending = self.pending_jobs.get(node_id, set())
        return len(pending) == 0

    def get_pending_job_count(self, node_id: str) -> int:
        return len(self.pending_jobs.get(node_id, set()))

    def get_completed_job_count(self, node_id: str) -> int:
        return len(self.completed_jobs.get(node_id, set()))

    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        return self.job_results.get(job_id)

    def get_failed_jobs(self, node_id: str) -> List[str]:
        failed = []
        for job_id in self.completed_jobs.get(node_id, set()):
            result = self.job_results.get(job_id)
            if result and not result.success:
                failed.append(job_id)
        return failed

    def reconstruct_from_job_queues(self, job_queues: "JobQueues") -> None:
        # Clear existing state
        self.pending_jobs.clear()
        self.completed_jobs.clear()
        self.job_results.clear()

        # Parse all jobs in the queues to reconstruct node relationships
        for procedure in ["upward", "downward"]:
            jobs = job_queues.get_jobs(procedure)
            for job_id, _ in jobs:
                # Parse job_id: format is {node_id}_{procedure}_{idx}
                try:
                    parts = job_id.rsplit('_', 2)  # Split from right, max 2 splits
                    if len(parts) >= 3:
                        node_id = parts[0]
                        # Add to pending jobs for this node
                        if node_id not in self.pending_jobs:
                            self.pending_jobs[node_id] = set()
                        if node_id not in self.completed_jobs:
                            self.completed_jobs[node_id] = set()
                        self.pending_jobs[node_id].add(job_id)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse job_id '{job_id}' during reconstruction")

        logger.info(f"Reconstructed JobTracker state for {len(self.pending_jobs)} nodes with {sum(len(jobs) for jobs in self.pending_jobs.values())} pending jobs")
