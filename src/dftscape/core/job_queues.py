from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from .search_primitives import RunCondition
from .graph import SearchableCriticalPointGraph


class JobQueues:

    def __init__(self) -> None:
        self._queues: Dict[str, List[Tuple[str, RunCondition]]] = {
            "upward": [],
            "downward": [],
        }

    def get_queue_length(self, procedure: str) -> int:
        return len(self._queues.get(procedure, []))

    def get_all_queue_lengths(self) -> Dict[str, int]:
        return {procedure: len(queue) for procedure, queue in self._queues.items()}

    def add_jobs(self, procedure: str, jobs: List[Tuple[str, RunCondition]]) -> None:
        if procedure not in self._queues:
            self._queues[procedure] = []
        self._queues[procedure].extend(jobs)

    def get_jobs(self, procedure: str) -> List[Tuple[str, RunCondition]]:
        return self._queues.get(procedure, [])

    def clear_queue(self, procedure: str) -> List[Tuple[str, RunCondition]]:
        jobs = self._queues.get(procedure, []).copy()
        if procedure in self._queues:
            self._queues[procedure].clear()
        return jobs

    def is_queue_empty(self, procedure: str) -> bool:
        return len(self._queues.get(procedure, [])) == 0

    def peek_job(self, procedure: str) -> Optional[Tuple[str, RunCondition]]:
        queue = self._queues.get(procedure, [])
        if queue:
            return queue[0]
        return None

    def remove_job(self, procedure: str, job_id: str) -> bool:
        queue = self._queues.get(procedure, [])
        for i, (jid, _) in enumerate(queue):
            if jid == job_id:
                queue.pop(i)
                return True
        return False

    def to_config(self) -> Dict[str, List[Tuple[str, Dict]]]:
        return {
            procedure: [
                (job_id, run_condition.to_config()) for job_id, run_condition in jobs
            ]
            for procedure, jobs in self._queues.items()
        }

    @classmethod
    def from_config(cls, config: Dict[str, List[Tuple[str, Dict]]]) -> "JobQueues":
        queues = cls()
        for procedure, jobs in config.items():
            queues._queues[procedure] = [
                (job_id, RunCondition.from_config(run_condition_config))
                for job_id, run_condition_config in jobs
            ]
        return queues

    def report(self, graph: Optional["SearchableCriticalPointGraph"] = None, verbose=False) -> str:
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Job Queues Status Summary")
        lines.append("=" * 80)

        total_pending_jobs = 0
        queue_lengths = self.get_all_queue_lengths()

        for procedure, length in queue_lengths.items():
            lines.append(f"  {procedure.capitalize()} pending jobs: {length}")
            total_pending_jobs += length

            # Show detailed job information if graph is available
            if graph is not None and length > 0 and verbose:
                jobs = self._queues.get(procedure, [])
                for job_id, run_condition in jobs:
                    # Extract node_id from job_id (format: node_id_procedure_idx)
                    parts = job_id.rsplit('_', 2)  # Split from right, max 2 splits
                    if len(parts) >= 3:
                        node_id = parts[0]
                        # Get the critical point data for this node
                        try:
                            critical_point = graph._graph.nodes[node_id]["data"]
                            morse_index = critical_point.index if critical_point else "?"
                        except (KeyError, AttributeError):
                            morse_index = "?"
                    else:
                        morse_index = "?"

                    # Format perturbation strength with sign
                    pert_str = f"{run_condition.perturbation_strength:+.3f}"

                    # Format perturbation direction (show first few elements)
                    pert_dir_str = f"[{run_condition.perturbation_direction[0]:.1f}"
                    if len(run_condition.perturbation_direction) > 1:
                        pert_dir_str += f", {run_condition.perturbation_direction[1]:.1f}"
                    if len(run_condition.perturbation_direction) > 2:
                        pert_dir_str += ", ..."
                    pert_dir_str += "]"

                    lines.append(f"    - Job {job_id}: index ({morse_index}), perturbation ({pert_str}), direction {pert_dir_str}")

        lines.append(f"  Total pending jobs: {total_pending_jobs}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        lengths = self.get_all_queue_lengths()
        return f"JobQueues(upward={lengths['upward']}, downward={lengths['downward']})"
