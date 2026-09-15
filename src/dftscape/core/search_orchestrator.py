from __future__ import annotations
from typing import List, Optional
import time
import logging

import numpy as np

from ..common import BaseRegistry
from .graph import SearchableCriticalPointGraph
from .search_primitives import RunCondition
from .point import CriticalPoint
from .search_strategies import (
    SearchStrategy,
    FirstNontrivialUpwardSearchStrategy,
    ExhaustiveDownwardSearchStrategy,
)
from .job_queues import JobQueues
from .job_tracker import JobTracker
from .node_state_manager import NodeStateManager
from .job_executor import JobExecutor
from .interfaces import HiSDRunner
from .search_monitor import SearchMonitor
from ..checkpoint.registry import CheckpointRegistry
from .callbacks import OrchestratorCallback

import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass
import hashlib

# Set up logging
logger = logging.getLogger(__name__)

# Gradient-projection validation constants
GRADIENT_TEST_MAGNITUDE_CAP = 0.5
GRADIENT_TEST_LINEAR_STEP = 0.05

# ------------------------------------------------------------------ #
#       SearchOrchestratorRegistry - Registry for orchestrators      #
# ------------------------------------------------------------------ #


class SearchOrchestratorRegistry(BaseRegistry["SearchOrchestrator"]):

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> "SearchOrchestrator":
        return super().create(name, *args, **kwargs)


# ------------------------------------------------------------------ #
#                SearchOrchestrator - Main coordinator               #
# ------------------------------------------------------------------ #


@SearchOrchestratorRegistry.register("serial")
class SearchOrchestrator:

    def __init__(
        self,
        graph: SearchableCriticalPointGraph,
        job_queues: JobQueues,
        hisd_runner: HiSDRunner,
        d_max: Optional[int] = None,
        n_directions: int = 1,
        perturbation_magnitude: float = 0.1,
        callbacks: Optional[List[OrchestratorCallback]] = None,
    ):
        if not isinstance(graph, SearchableCriticalPointGraph):
            raise TypeError("graph must be a SearchableCriticalPointGraph instance")
        if not isinstance(job_queues, JobQueues):
            raise TypeError("job_queues must be a JobQueues instance")
        if d_max is not None and d_max < 0:
            raise ValueError("d_max must be non-negative or None")
        if n_directions < 1:
            raise ValueError("n_directions must be at least 1")
        if perturbation_magnitude <= 0:
            raise ValueError("perturbation_magnitude must be positive")

        self.graph = graph
        self.job_queues = job_queues
        self.d_max = d_max
        self.n_directions = n_directions
        self.job_tracker = JobTracker()
        self.node_states = NodeStateManager(graph)
        self.executor = JobExecutor(hisd_runner, max_concurrent=1)  # Start with serial

        # Search parameters
        self.perturbation_magnitude = perturbation_magnitude

        # Use composition with strategies (configured with default perturbation magnitude)
        self.upward_strategy = FirstNontrivialUpwardSearchStrategy(
            perturbation_magnitude=self.perturbation_magnitude
        )
        self.downward_strategy = ExhaustiveDownwardSearchStrategy(
            perturbation_magnitude=self.perturbation_magnitude
        )

        # Callbacks for job batch completion
        self.callbacks = callbacks

        # Statistics and monitoring
        self.monitor: SearchMonitor = SearchMonitor()

    def run_upward_search(self) -> None:
        self.run_search(self.upward_strategy, "upward")

    def run_search(self, strategy: "SearchStrategy", procedure: str) -> None:
        self._initialize_search(procedure)

        while not self._is_search_finished(procedure):
            # For each unprocessed node, ask the strategy which RunConditions to spawn
            unprocessed_nodes = self.node_states.get_unprocessed_nodes(procedure)
            logger.debug(f"Found {len(unprocessed_nodes)} unprocessed nodes")

            for node_id in unprocessed_nodes:
                if node_id not in self.graph._graph.nodes:
                    logger.warning(f"Node {node_id} not found in graph")
                    self.node_states.mark_processed(node_id, procedure)
                    continue

                # Use helper method to spawn jobs (handles all strategy delegation and job ID creation)
                jobs = self._spawn_jobs_from_node(node_id, procedure)

                # If no jobs were spawned for this node, mark it as processed immediately
                if not jobs:
                    self.node_states.mark_processed(node_id, procedure)
                    logger.debug(
                        f"Node {node_id} marked processed immediately (no jobs spawned)"
                    )
                else:
                    # Add to queues and tracker, then mark node as processing
                    self.job_queues.add_jobs(procedure, jobs)
                    job_ids = [job_id for job_id, _ in jobs]
                    self.job_tracker.add_jobs_for_node(node_id, job_ids)
                    self.node_states.mark_processing(node_id, procedure)
                    logger.debug(
                        f"Node {node_id} marked as processing with {len(jobs)} jobs spawned"
                    )

            # Execute pending jobs and log progress
            self._execute_pending_jobs(procedure)
            self._log_progress(procedure)

        self._finalize_search(procedure)

    def run_downward_search(self) -> None:
        # Step 1: Synchronization check
        if not self.is_upward_search_complete():
            raise RuntimeError(
                "Cannot start downward search: upward search is not complete"
            )

        # Step 2: preserve any loaded node state (no re-initialization)
        self.run_search(self.downward_strategy, "downward")

    def run_full_search(self) -> None:
        logger.debug("Starting full search (upward + downward)")

        # Phase 1: Upward search
        self.run_upward_search()

        # Phase 2: Downward search (only if upward is complete)
        if self.is_upward_search_complete():
            self.run_downward_search()
        else:
            logger.error("Cannot proceed to downward search: upward search incomplete")

        logger.debug("Full search completed")

    def _initialize_search(self, procedure: str) -> None:
        logger.debug(f"Starting {procedure} search")
        if procedure == "upward" and self.d_max is not None:
            logger.debug(f"Maximum search order (d_max): {self.d_max}")

        # Start monitoring the search
        self.monitor.start_search(procedure)

    def _is_search_finished(self, procedure: str) -> bool:
        queue_empty = self.job_queues.is_queue_empty(procedure)
        has_unprocessed = self.node_states.has_unprocessed_nodes(procedure)
        processing_count = len(self.node_states.get_processing_nodes(procedure))

        logger.debug(
            f"DEBUG _is_search_finished({procedure}): queue_empty={queue_empty}, has_unprocessed={has_unprocessed}, processing_count={processing_count}"
        )

        finished = queue_empty and not has_unprocessed and processing_count == 0
        if not finished:
            logger.debug(
                f"DEBUG: Search not finished - queue_empty={queue_empty}, unprocessed={has_unprocessed}, processing={processing_count}"
            )

        return finished

    def _execute_pending_jobs(self, procedure: str) -> None:
        while not self.job_queues.is_queue_empty(procedure):
            # -----------------------------------------------------------------
            # DYNAMIC PRIORITY SORTING
            # Sort the queue so lowest index (k) is at the front.
            # We only do this for downward searches to prioritize grounding the network.
            # -----------------------------------------------------------------
            if procedure == "downward" and hasattr(self.job_queues, "_queues"):
                try:
                    # _queues is a list of tuples: [(job_id, run_condition), ...]
                    # We sort in-place based on the run_condition's index
                    self.job_queues._queues[procedure].sort(key=lambda item: item[1].index)
                except Exception as e:
                    logger.debug(f"Could not sort downward queue: {e}")
            # -----------------------------------------------------------------
    

            # Peek at the next job without removing it
            job_info = self.job_queues.peek_job(procedure)
            if job_info is None:
                break

            job_id, run_condition = job_info
            
            # pass the job_id to dynamics.py
            os.environ["HISD_JOB_ID"] = "serial_job"

            # Execute the job
            result = self.executor.execute_job(run_condition, job_id)

            # Record job completion in monitor
            self.monitor.record_job_completion(
                procedure=procedure,
                job_id=job_id,
                success=result.success,
                execution_time=result.execution_time,
                node_id=(
                    run_condition.node_id
                    if hasattr(run_condition, "node_id")
                    else job_id
                ),
            )

            # Handle the result - wrap in try/catch to ensure job is always removed
            self._handle_job_result(result, procedure)

            # Remove the job from queue since algorithm is deterministic
            # Whether successful or failed, retrying won't help
            self.job_queues.remove_job(procedure, job_id)

            # Trigger callback after job is fully processed and removed
            # This ensures the job queue state is updated when the callback runs
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_state_change(self)

    def _handle_job_result(self, result, procedure: str) -> None:
        if result.success and result.critical_point is not None:
            # Handle successful job
            # Handle fragmented molecules first; we don't spawn any jobs from those
            if result.critical_point.num_fragments > 1:
                logger.info(f"Job {result.job_id} resulted in a fragmented structure. Terminating pathway.")
                
                # find or add the target node (Fragment-aware RMSD handles duplicates here)
                target_node_id = self._find_or_add_target_node(result.critical_point)

                # get the source node that spawned this job
                source_node_id = self._get_parent_node_id(
                    result.run_condition, result.job_id
                )
    
                # add edge between source and target
                if result.add_edge:
                    edge_changed = self._add_edge_between_nodes(
                        source_node_id, target_node_id, result.run_condition
                    )
                
                # mark the fragmented target as completely processed so nothing ever spawns from it
                self.node_states.mark_processed(target_node_id, "upward")
                self.node_states.mark_processed(target_node_id, "downward")

                # mark the specific job as complete!
                self.job_tracker.mark_job_complete(
                    source_node_id, result.job_id, result
                )
                
                # check if the source node that spawned this is now completely finished
                self._check_and_mark_node_processed(source_node_id, procedure)
            elif procedure == "upward":
                # Add to graph (handles duplicates)
                was_added = self.graph.add(result.critical_point)
                # Find or add the target node
                target_node_id = self._find_or_add_target_node(result.critical_point)
                # Get the source node that spawned this job
                source_node_id = self._get_parent_node_id(
                    result.run_condition, result.job_id
                )
                
                # add edge between source and target
                if result.add_edge:
                    edge_changed = self._add_edge_between_nodes(
                        source_node_id, target_node_id, result.run_condition
                    )
                # Only spawn jobs from new node if it's truly unprocessed
                # IMPORTANT: Check node state to avoid race condition where multiple jobs
                # discover the same critical point and both try to spawn jobs from it
                if was_added:
                    new_node_id = result.critical_point.xyz
                    new_node_is_unprocessed = not self.node_states.is_processing(
                        new_node_id, "upward"
                    ) and not self.node_states.is_processed(new_node_id, "upward")

                    if new_node_is_unprocessed:
                        # Spawn new jobs from the new node
                        new_jobs = self._spawn_jobs_from_node(new_node_id, "upward")
                        if new_jobs:
                            # Register jobs in tracker
                            job_ids = [job_id for job_id, _ in new_jobs]
                            self.job_tracker.add_jobs_for_node(new_node_id, job_ids)
                            # Add to queue
                            self.job_queues.add_jobs("upward", new_jobs)
                            # Mark new node as processing when jobs are spawned
                            self.node_states.mark_processing(new_node_id, "upward")
                            logger.debug(
                                f"Added new node {new_node_id} with {len(new_jobs)} new jobs and marked as processing"
                            )

                # Mark job as complete immediately after HiSD run (for bookkeeping)
                parent_node_id = self._get_parent_node_id(
                    result.run_condition, result.job_id
                )
                logger.debug(
                    f"DEBUG: Marking job complete - parent_node_id='{parent_node_id}', job_id='{result.job_id}'"
                )
                self.job_tracker.mark_job_complete(
                    parent_node_id, result.job_id, result
                )

                # Check if all jobs for this node are complete and mark node processed
                self._check_and_mark_node_processed(parent_node_id, procedure)
            elif procedure == "downward":
                # Find or add the target node
                target_node_id = self._find_or_add_target_node(result.critical_point)

                # Get the source node that spawned this job
                source_node_id = self._get_parent_node_id(
                    result.run_condition, result.job_id
                )

                # Add edge between source and target (always, regardless of whether target was new)
                if result.add_edge:
                    edge_changed = self._add_edge_between_nodes(
                        source_node_id, target_node_id, result.run_condition
                    )

                # Track if jobs were added to queue
                jobs_added = False

                # IMPORTANT: Check node state instead of comparing IDs to avoid race condition
                # where multiple jobs discover the same node simultaneously and both try to spawn jobs
                target_is_unprocessed = not self.node_states.is_processing(
                    target_node_id, "downward"
                ) and not self.node_states.is_processed(target_node_id, "downward")

                if target_is_unprocessed:
                    # ONLY spawn downward jobs here. 
                    # The master script's cycle loop will naturally pick up this node 
                    # and spawn upward jobs for it during the next upward phase!
                    new_jobs = self._spawn_jobs_from_node(target_node_id, "downward")
                    if new_jobs:
                        # Register jobs in tracker
                        job_ids = [job_id for job_id, _ in new_jobs]
                        self.job_tracker.add_jobs_for_node(target_node_id, job_ids)
                        # Add to queue
                        self.job_queues.add_jobs("downward", new_jobs)
                        # Mark new node as processing when jobs are spawned
                        self.node_states.mark_processing(target_node_id, "downward")
                        logger.debug(
                            f"Added new node {target_node_id} with {len(new_jobs)} new jobs and marked as processing"
                        )
                        jobs_added = True

                # Mark job as complete immediately after HiSD run (for bookkeeping)
                logger.debug(
                    f"DEBUG: Marking job complete - source_node_id='{source_node_id}', job_id='{result.job_id}'"
                )
                self.job_tracker.mark_job_complete(
                    source_node_id, result.job_id, result
                )

                # Check if all jobs for this node are complete and mark node processed
                self._check_and_mark_node_processed(source_node_id, procedure)

            else:
                raise ValueError(f"Unknown procedure: {procedure}")

        else:
            # Handle failed job (bookkeeping only)
            logger.warning(f"Job {result.job_id} failed: {result.error}")
            parent_node_id = self._get_parent_node_id(
                result.run_condition, result.job_id
            )
            logger.debug(
                f"DEBUG: Marking failed job complete - parent_node_id='{parent_node_id}', job_id='{result.job_id}'"
            )
            self.job_tracker.mark_job_complete(parent_node_id, result.job_id, result)

            # Check if all jobs for this node are complete and mark node processed
            self._check_and_mark_node_processed(parent_node_id, procedure)

    def _dump_node_job_state(self, node_id: str) -> str:
        try:
            pending_set = self.job_tracker.pending_jobs.get(node_id, set())
            completed_set = self.job_tracker.completed_jobs.get(node_id, set())
            # Show first few job IDs from each set
            pending_list = sorted(list(pending_set))[:3]
            completed_list = sorted(list(completed_set))[:3]
            return f"pending_jobs={pending_list} completed_jobs={completed_list}"
        except Exception as e:
            return f"<error reading job tracker: {e}>"

    def _log_progress(self, procedure: str) -> None:
        stats = self.node_states.get_processing_stats(procedure)
        search_stats = self.monitor.search_stats.get(procedure)
        jobs_executed = search_stats.total_jobs if search_stats else 0
        queued = self.job_queues.get_queue_length(procedure)

        # Basic progress message
        logger.debug(
            f"Progress: {stats['processed']}/{stats['total_nodes']} nodes processed, "
            f"{queued} jobs queued, {jobs_executed} jobs executed"
        )

        # Sanity/invariant checks: processed + processing + unprocessed should equal total
        processed = stats.get("processed", 0)
        processing = stats.get("processing", 0)
        unprocessed = stats.get("unprocessed", 0)
        total_nodes = stats.get("total_nodes", 0)

        if processed + processing + unprocessed != total_nodes:
            logger.warning(
                "Node state counts mismatch: processed(%d) + processing(%d) + unprocessed(%d) != total_nodes(%d)",
                processed,
                processing,
                unprocessed,
                total_nodes,
            )

            # Dump detailed per-node diagnostics for a few nodes in each category
            try:
                processing_nodes = self.node_states.get_processing_nodes(procedure)
                unprocessed_nodes = set(
                    self.node_states.get_unprocessed_nodes(procedure)
                )
            except Exception as e:
                logger.exception("Failed to fetch node lists for diagnostics: %s", e)
                processing_nodes = []
                unprocessed_nodes = set()

            # Show up to 5 processing nodes
            for nid in list(processing_nodes)[:5]:
                try:
                    pending = self.job_tracker.get_pending_job_count(nid)
                except Exception:
                    pending = "<err>"
                try:
                    completed = self.job_tracker.get_completed_job_count(nid)
                except Exception:
                    completed = "<err>"
                logger.warning(
                    "Processing node diagnostic: %s pending_jobs=%s completed_jobs=%s tracker_dump=%s",
                    nid[:120],
                    pending,
                    completed,
                    self._dump_node_job_state(nid),
                )

            # Show up to 5 unprocessed nodes (they should have no pending/completed jobs)
            for nid in list(unprocessed_nodes)[:5]:
                try:
                    pending = self.job_tracker.get_pending_job_count(nid)
                except Exception:
                    pending = "<err>"
                try:
                    completed = self.job_tracker.get_completed_job_count(nid)
                except Exception:
                    completed = "<err>"
                logger.warning(
                    "Unprocessed node diagnostic: %s pending_jobs=%s completed_jobs=%s tracker_dump=%s",
                    nid[:120],
                    pending,
                    completed,
                    self._dump_node_job_state(nid),
                )

        # If there are processing nodes but zero queued jobs, that's suspicious
        if processing > 0 and queued == 0:
            logger.warning(
                "There are %d processing nodes but job queue length is 0 for procedure '%s'",
                processing,
                procedure,
            )
            # Dump queue status and a couple of processing nodes
            try:
                logger.debug(
                    "JobQueues snapshot: %s", self.job_queues.get_jobs(procedure)[:5]
                )
            except Exception:
                logger.debug("JobQueues snapshot: <error obtaining jobs>")
            # Also dump per-processing-node job-tracker diagnostics to help root-cause
            try:
                processing_nodes = self.node_states.get_processing_nodes(procedure)
                if processing_nodes:
                    logger.debug(
                        "Processing nodes (count=%d): %s",
                        len(processing_nodes),
                        processing_nodes[:10],
                    )
                    for nid in list(processing_nodes)[:10]:
                        try:
                            pending = self.job_tracker.get_pending_job_count(nid)
                        except Exception:
                            pending = "<err>"
                        try:
                            completed = self.job_tracker.get_completed_job_count(nid)
                        except Exception:
                            completed = "<err>"
                        try:
                            tracker_dump = self._dump_node_job_state(nid)
                        except Exception:
                            tracker_dump = "<err>"
                        logger.warning(
                            "Processing-node diagnostic (proc=%s): %s pending=%s completed=%s tracker_dump=%s",
                            procedure,
                            nid[:120],
                            pending,
                            completed,
                            tracker_dump,
                        )
            except Exception:
                logger.exception("Failed to emit per-node job-tracker diagnostics")

    def _finalize_search(self, procedure: str) -> None:
        # End monitoring the search
        self.monitor.end_search(procedure)

        # Get final statistics for logging
        stats = self.monitor.search_stats.get(procedure)
        if stats:
            logger.debug(
                f"{procedure.capitalize()} search completed in {stats.duration:.2f}s, "
                f"{stats.total_jobs} jobs executed, {stats.completed_jobs} successful, "
                f"{stats.failed_jobs} failed"
            )

    def is_upward_search_complete(self) -> bool:
        upward_queue_empty = self.job_queues.is_queue_empty("upward")
        all_nodes_upward_processed = not self.node_states.has_unprocessed_nodes(
            "upward"
        )

        return upward_queue_empty and all_nodes_upward_processed

    def is_downward_search_complete(self) -> bool:
        downward_queue_empty = self.job_queues.is_queue_empty("downward")
        all_nodes_downward_processed = not self.node_states.has_unprocessed_nodes(
            "downward"
        )

        return downward_queue_empty and all_nodes_downward_processed

    def _check_and_mark_node_processed(self, node_id: str, procedure: str) -> None:
        # Get all currently pending jobs for this node directly from the tracker
        pending_jobs = self.job_tracker.pending_jobs.get(node_id, set())
        
        # Check if there are any pending jobs specifically for THIS procedure
        procedure_marker = f"_{procedure}_"
        pending_for_procedure = any(procedure_marker in jid for jid in pending_jobs)
        
        # If no jobs are left for this specific procedure, it is safe to mark it processed
        if not pending_for_procedure:
            self.node_states.mark_processed(node_id, procedure)
            logger.debug(
                f"✓ Node {node_id} fully processed for '{procedure}' immediately after job completion"
            )

    # ------------------------------------------------------------------ #
    #                 GRADIENT PROJECTION VALIDATION                     #
    # ------------------------------------------------------------------ #
    #
    # Both FirstNontrivialUpwardSearchStrategy and
    # ExhaustiveDownwardSearchStrategy spawn RunConditions in +v/-v pairs
    # for each unstable mode (see the master test/debug script). Before a
    # pair is committed as jobs, we sanity-check that the force -
    # projected onto the perturbation direction - has opposite sign at
    # +v and -v, i.e. the pair actually straddles the ridge/valley rather
    # than both landing on the same side of it.
    #
    # If a pair fails the test, we grow the perturbation magnitude for
    # that specific mode (min(current * 2, current + 0.05) each attempt)
    # and re-test, until it passes or the magnitude would exceed 0.5. If
    # we hit that cap without passing, we fall back to the strategy's
    # original (base) magnitude and only keep the +v job for that mode -
    # no -v job is spawned.
    # ------------------------------------------------------------------ #

    def _gradient_projection_test(
        self, rc_plus: RunCondition, rc_minus: RunCondition
    ) -> tuple[bool, float, float]:
        backend = self.executor.hisd_runner.backend

        v = rc_plus.perturbation_direction
        state_plus = rc_plus.get_perturbed_state()
        state_minus = rc_minus.get_perturbed_state()

        f_plus = -backend.compute_gradient(state_plus)
        f_minus = -backend.compute_gradient(state_minus)

        proj_plus = float(np.dot(f_plus, v))
        proj_minus = float(np.dot(f_minus, v))

        passed = proj_plus * proj_minus < 0
        return passed, proj_plus, proj_minus

    def _respawn_pair_at_magnitude(
        self,
        node_data,
        strategy: "SearchStrategy",
        mode_idx,
        magnitude: float,
    ) -> tuple[Optional[RunCondition], Optional[RunCondition]]:
        """
        Re-runs strategy.spawn_jobs at a given perturbation_magnitude and
        pulls out the +v/-v pair matching mode_idx. Other modes produced
        by this respawn are discarded - the caller already holds the
        validated jobs for those modes from earlier in the loop.
        """
        original_magnitude = getattr(strategy, "perturbation_magnitude", None)
        try:
            strategy.perturbation_magnitude = magnitude
            run_conditions = strategy.spawn_jobs(node_data)
        except Exception as e:
            logger.error(
                f"Failed to respawn jobs at perturbation_magnitude={magnitude}: {e}"
            )
            return None, None
        finally:
            if original_magnitude is not None:
                strategy.perturbation_magnitude = original_magnitude

        for i in range(0, len(run_conditions) - 1, 2):
            if getattr(run_conditions[i], "perturbation_idx", None) == mode_idx:
                return run_conditions[i], run_conditions[i + 1]

        logger.warning(
            f"Could not locate mode {mode_idx} when respawning at magnitude {magnitude}"
        )
        return None, None

    def _verify_and_adjust_run_conditions(
        self,
        node_data,
        strategy: "SearchStrategy",
        run_conditions: List[RunCondition],
    ) -> List[RunCondition]:
        base_magnitude = getattr(strategy, "perturbation_magnitude", None)
        if base_magnitude is None:
            # Strategy doesn't expose a tunable perturbation_magnitude;
            # nothing we can safely adjust, so pass jobs through unmodified.
            return run_conditions

        verified: List[RunCondition] = []
        i = 0
        n = len(run_conditions)

        while i < n:
            rc_plus = run_conditions[i]

            # Only treat consecutive entries as a testable +v/-v pair if
            # they share the same mode index.
            is_pair = (
                i + 1 < n
                and getattr(run_conditions[i + 1], "perturbation_idx", None)
                == getattr(rc_plus, "perturbation_idx", None)
            )

            if not is_pair:
                verified.append(rc_plus)
                i += 1
                continue

            rc_minus = run_conditions[i + 1]
            mode_idx = rc_plus.perturbation_idx
            current_magnitude = base_magnitude

            while True:
                try:
                    passed, proj_plus, proj_minus = self._gradient_projection_test(
                        rc_plus, rc_minus
                    )
                except Exception as e:
                    logger.error(
                        f"Gradient projection test errored for mode {mode_idx}: {e}. "
                        "Keeping pair unvalidated."
                    )
                    verified.extend([rc_plus, rc_minus])
                    break

                logger.debug(
                    f"Gradient projection test (mode={mode_idx}, "
                    f"magnitude={current_magnitude:.6f}): "
                    f"proj_plus={proj_plus:+.6e}, proj_minus={proj_minus:+.6e}, "
                    f"passed={passed}"
                )

                if passed:
                    verified.extend([rc_plus, rc_minus])
                    break

                next_magnitude = min(
                    current_magnitude * 2, current_magnitude + GRADIENT_TEST_LINEAR_STEP
                )

                if next_magnitude > GRADIENT_TEST_MAGNITUDE_CAP:
                    logger.warning(
                        f"Mode {mode_idx} failed the gradient projection test up to "
                        f"magnitude {current_magnitude:.6f}; reverting to base "
                        f"magnitude {base_magnitude:.6f} and spawning a single job "
                        "instead of a +v/-v pair."
                    )
                    rc_reverted, _ = self._respawn_pair_at_magnitude(
                        node_data, strategy, mode_idx, base_magnitude
                    )
                    verified.append(rc_reverted if rc_reverted is not None else rc_plus)
                    break

                new_rc_plus, new_rc_minus = self._respawn_pair_at_magnitude(
                    node_data, strategy, mode_idx, next_magnitude
                )
                if new_rc_plus is None or new_rc_minus is None:
                    # Respawn failed; keep the last successfully spawned pair
                    verified.extend([rc_plus, rc_minus])
                    break

                rc_plus, rc_minus = new_rc_plus, new_rc_minus
                current_magnitude = next_magnitude

            i += 2

        return verified

    def _spawn_jobs_from_node(
        self, node_id: str, procedure: str
    ) -> List[tuple[str, RunCondition]]:
        if node_id not in self.graph._graph.nodes:
            logger.warning(f"Node {node_id} not found in graph")
            return []

        node_data = self.graph._graph.nodes[node_id]["data"]

        # Check d_max limit for upward search (orchestrator still authoritative)
        if procedure == "upward" and self.d_max is not None:
            if node_data.index >= self.d_max:
                logger.debug(
                    f"Node {node_id} has index {node_data.index} >= d_max ({self.d_max}), skipping upward job spawning"
                )
                return []

        # Choose strategy for this procedure
        if procedure == "upward":
            strategy = self.upward_strategy
        elif procedure == "downward":
            strategy = self.downward_strategy
        else:
            logger.error(f"Unknown procedure: {procedure}")
            return []

        try:
            run_conditions = strategy.spawn_jobs(node_data)
        except Exception as e:
            logger.error(f"Strategy.spawn_jobs failed for node {node_id}: {e}")
            return []

        if not run_conditions:
            return []

        # Validate +v/-v pairs against the gradient projection test, growing
        # the perturbation magnitude where needed (see docstring above).
        try:
            run_conditions = self._verify_and_adjust_run_conditions(
                node_data, strategy, run_conditions
            )
        except Exception as e:
            logger.error(
                f"Gradient projection validation failed for node {node_id}, "
                f"falling back to unvalidated run conditions: {e}"
            )

        # Create job ids and return job tuples
        jobs: List[tuple[str, RunCondition]] = []
        for idx, rc in enumerate(run_conditions):
            job_id = f"{node_id}_{procedure}_{idx}"
            jobs.append((job_id, rc))

        # NOTE: Job tracking is done by the caller (run_search) to avoid duplicate registration
        return jobs

    def _get_parent_node_id(self, run_condition: RunCondition, job_id: str) -> str:
        # Extract parent node ID from job_id by finding the last occurrence of '_upward_' or '_downward_'
        # and taking everything before that
        if "_upward_" in job_id:
            parent_node_id = job_id.rsplit("_upward_", 1)[0]
        elif "_downward_" in job_id:
            parent_node_id = job_id.rsplit("_downward_", 1)[0]
        else:
            # Fallback: use the XYZ from the run condition state
            logger.warning(
                f"Could not extract parent node ID from job_id {job_id}, falling back to XYZ computation"
            )
            parent_xyz = self.executor.hisd_runner.backend.get_xyz(run_condition.state)
            return parent_xyz

        return parent_node_id

    def _find_or_add_target_node(self, critical_point: CriticalPoint) -> str:
        # Try to add the critical point (this will handle duplicates via distance)
        was_added = self.graph.add(critical_point)

        if was_added:
            # It was a new node
            return critical_point.xyz
        else:
            # It was a duplicate, find the existing node
            # We need to search for a node with matching distance
            target_node_id = None
            for node_id, node_data in self.graph._graph.nodes(data=True):
                if (
                    self.graph.distance_function(
                        critical_point,
                        node_data["data"],
                        **self.graph.distance_kwargs,
                    ) #type: ignore
                    < self.graph.threshold
                ):
                    target_node_id = node_id
                    break

            if target_node_id is None:
                logger.error(
                    "Could not find existing node for duplicate critical point"
                )
                # Fallback: use the XYZ as ID anyway
                target_node_id = critical_point.xyz

            return target_node_id

    def _add_edge_between_nodes(
        self, source_node_id: str, target_node_id: str, run_condition: RunCondition
    ) -> bool:
        # Check if edge already exists and get its current data
        edge_exists = self.graph._graph.has_edge(source_node_id, target_node_id)
        old_edge_data = None
        if edge_exists:
            old_edge_data = self.graph._graph.edges[
                source_node_id, target_node_id
            ].copy()

        # Add edge to the graph with metadata
        edge_data = {
            "run_condition": run_condition,
            "direction": "downward",  # This is a downward transition
            "created_at": time.time(),
        }

        self.graph._graph.add_edge(source_node_id, target_node_id, **edge_data)

        # Check if the edge data actually changed
        new_edge_data = self.graph._graph.edges[source_node_id, target_node_id]
        data_changed = self._compare_edge_data(old_edge_data, new_edge_data)

        if data_changed:
            logger.debug(
                f"Added/updated downward edge from {source_node_id} to {target_node_id}"
            )
        else:
            logger.debug(
                f"Edge from {source_node_id} to {target_node_id} already exists with same data"
            )

        return data_changed

    def _compare_edge_data(self, old_data: dict, new_data: dict) -> bool:
        if old_data is None and new_data is None:
            return False
        if old_data is None or new_data is None:
            return True
        if set(old_data.keys()) != set(new_data.keys()):
            return True

        # Fix: exclude transient metadata from comparison
        EXCLUDE_FROM_COMPARISON = {"created_at"}
            
        for key in old_data.keys():
            if key in EXCLUDE_FROM_COMPARISON:
                continue
            old_value = old_data[key] 
            new_value = new_data[key]

            # Handle numpy arrays
            if hasattr(old_value, "shape") and hasattr(new_value, "shape"):
                try:
                    import numpy as np

                    if not np.array_equal(old_value, new_value):
                        return True
                except ImportError:
                    # Fallback if numpy not available
                    if not (old_value == new_value).all():
                        return True
            # Handle other array-like objects
            elif hasattr(old_value, "__len__") and hasattr(new_value, "__len__"):
                try:
                    if len(old_value) != len(new_value):
                        return True
                    # Try element-wise comparison
                    if not all(a == b for a, b in zip(old_value, new_value)):
                        return True
                except (TypeError, ValueError):
                    # If comparison fails, assume they're different
                    return True
            # Handle regular values
            else:
                try:
                    if old_value != new_value:
                        return True
                except ValueError:
                    # If comparison fails (e.g., ambiguous array truth value), assume different
                    return True

        return False

    def report(self) -> str:
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("ORCHESTRATOR SEARCH STATUS SUMMARY")
        lines.append("=" * 80)

        # Monitor status (includes search stats and node processing)
        lines.append("\n" + self.monitor.report(self))

        # Graph status
        lines.append("\n" + self.graph.report())

        # Job queue status (not covered by monitor or graph reports)
        lines.append("\n" + self.job_queues.report(self.graph))

        lines.append("=" * 80)

        return "\n".join(lines)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        graph: SearchableCriticalPointGraph,
        hisd_runner: HiSDRunner,
        callbacks: Optional[List[OrchestratorCallback]] = None,
    ) -> "SearchOrchestrator":
        # Load checkpoint data using JSON checkpointer
        json_checkpointer = CheckpointRegistry.create("json")
        checkpoint_data = json_checkpointer.load(checkpoint_path)

        # Restore graph from checkpoint
        if "graph" in checkpoint_data:
            graph = SearchableCriticalPointGraph.from_config(checkpoint_data["graph"])

        # Create new orchestrator instance
        orchestrator = cls(
            graph=graph,
            job_queues=JobQueues(),
            hisd_runner=hisd_runner,
            d_max=checkpoint_data.get("d_max"),
            perturbation_magnitude=checkpoint_data.get("perturbation_magnitude", 0.1),
            callbacks=callbacks,
        )

        # Restore job queues using JobQueues.from_config if present
        raw_job_queues = checkpoint_data.get(
            "job_queues", {"upward": [], "downward": []}
        )
        try:
            orchestrator.job_queues = JobQueues.from_config(raw_job_queues)
        except Exception:
            # Fallback: manually reconstruct using RunCondition
            from .search_primitives import RunCondition

            orchestrator.job_queues = JobQueues()
            for procedure, jobs in raw_job_queues.items():
                reconstructed = [
                    (job_id, RunCondition.from_config(run_condition_config))
                    for job_id, run_condition_config in jobs
                ]
                orchestrator.job_queues._queues[procedure] = reconstructed

        return orchestrator




# A simple mock object to mimic the JobExecutor's return format
@dataclass
class JobResult:
    job_id: str
    run_condition: RunCondition
    success: bool
    error: Optional[str]
    critical_point: Optional[CriticalPoint]
    execution_time: float
    add_edge: bool


@SearchOrchestratorRegistry.register("parallel")
class ParallelSearchOrchestrator(SearchOrchestrator):
    def __init__(self, *args, max_concurrent: int = 50, job_dir: str = "pbs_jobs", base_cfg: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_concurrent = max_concurrent
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.base_cfg = base_cfg  # The global dict config to pass to workers
        self.running_jobs = {}    # job_id -> dict {"procedure": procedure, "run_condition": rc}

    def run_search(self, strategy: "SearchStrategy", procedure: str) -> None:
        self._initialize_search(procedure)

        # recover jobs once per run
        if not getattr(self, "_recovery_done", False):
            self._recover_active_jobs()
            self._recovery_done = True

        # Loop until queue is empty AND no jobs are currently running on PBS
        while not self._is_search_finished(procedure) or len(self.running_jobs) > 0:
            # harvest completed files
            self._collect_results()
            
            # eplicitly spawn and track jobs inline (just like the Serial orchestrator)
            unprocessed_nodes = self.node_states.get_unprocessed_nodes(procedure)
            logger.debug(f"Found {len(unprocessed_nodes)} unprocessed nodes")

            for node_id in unprocessed_nodes:
                if node_id not in self.graph._graph.nodes:
                    logger.warning(f"Node {node_id} not found in graph")
                    self.node_states.mark_processed(node_id, procedure)
                    continue

                jobs = self._spawn_jobs_from_node(node_id, procedure)

                if not jobs:
                    self.node_states.mark_processed(node_id, procedure)
                    logger.debug(
                        f"Node {node_id} marked processed immediately (no jobs spawned)"
                    )
                else:
                    self.job_queues.add_jobs(procedure, jobs)
                    job_ids = [job_id for job_id, _ in jobs]
                    self.job_tracker.add_jobs_for_node(node_id, job_ids)
                    self.node_states.mark_processing(node_id, procedure)
                    logger.debug(
                        f"Node {node_id} marked as processing with {len(jobs)} jobs spawned"
                    )

            self._log_progress(procedure)
            
            # submit jobs up to our PBS limit
            self._submit_pending_jobs(procedure)
                
            # sleep to respect the cluster scheduler and filesystem
            if len(self.running_jobs) > 0:
                logger.info(f"Waiting on {len(self.running_jobs)} PBS jobs. Sleeping 300s...")
                time.sleep(300)

        self._finalize_search(procedure)

    def _submit_pending_jobs(self, procedure: str) -> None:
        while len(self.running_jobs) < self.max_concurrent and not self.job_queues.is_queue_empty(procedure):
            job_info = self.job_queues.peek_job(procedure)
            if job_info is None:
                break
                
            job_id, run_condition = job_info
            self.job_queues.remove_job(procedure, job_id)
            success = self._submit_single_pbs_job(job_id, run_condition, procedure)

            if not success:
                logger.warning(f"Simulating failure result for un-submittable job {job_id}")
                failed_result = JobResult(
                    job_id=job_id,
                    run_condition=run_condition,
                    success=False,
                    error="Failed to submit job to PBS queue",
                    critical_point=None,
                    execution_time=0.0,
                    add_edge=False
                )
                self._handle_job_result(failed_result, procedure)
            

    def _get_active_pbs_jobs(self) -> set[str]:
        """Queries the cluster to get a set of all currently queued/running PBS Job IDs."""
        active_ids = set()
        try:
            # Query only this user's jobs to save parsing overhead
            res = subprocess.run(["qstat", "-f", "-F", "json"], capture_output=True, text=True)
            data = json.loads(res.stdout)
            active_ids = set(data.get("Jobs", {}).keys())
                        
        except Exception as e:
            logger.warning(f"Failed to query qstat: {e}. Assuming all jobs are still running.")
            # If qstat randomly fails, assume everything is still running to prevent false crashes
            return {info.get("pbs_id", "") for info in self.running_jobs.values()}
            
        return active_ids

    def _submit_single_pbs_job(self, job_id: str, run_condition: RunCondition, procedure: str) -> bool:
    
        # Generate a short, safe filename using MD5
        # Maps a 500+ character XYZ string into a strict 16-char ID
        job_hash = hashlib.md5(job_id.encode('utf-8')).hexdigest()[:16]
        safe_filename = f"job_{procedure}_{job_hash}"
        
        input_file = self.job_dir / f"{safe_filename}_in.json"
        output_file = self.job_dir / f"{safe_filename}_out.json"
        pbs_script = self.job_dir / f"{safe_filename}.pbs"

        # Extract the actual XYZ string from the raw job_id
        # and pass it explicitly so the backend can parse the elements
        parent_xyz = job_id.rsplit(f"_{procedure}_", 1)[0]

        #  write the payload
        payload = {
            "cfg": self.base_cfg,
            "run_condition": run_condition.to_config(),
            "parent_xyz": parent_xyz 
        }
        with open(input_file, 'w') as f:
            json.dump(payload, f)

        # Write the PBS Bash Script
        # Customize the header lines here based on your specific HPC cluster rules!
        pbs_content = f"""#!/bin/bash
#PBS -N {safe_filename}
#PBS -P 11004368 
#PBS -l ncpus=1
#PBS -l mem=16GB 
#PBS -l walltime=2:05:00
#PBS -o {self.job_dir}/{safe_filename}.log
#PBS -j oe

conda deactivate
source $HOME/dir/dftscape-dxtb-main/.venv/bin/activate

cd $PBS_O_WORKDIR

# export the safe filename as the job ID for dynamics.py
export HISD_JOB_ID="{safe_filename}"

python worker.py {input_file} {output_file}
"""
        with open(pbs_script, 'w') as f:
            f.write(pbs_content)

        # submit to PBS Queue
        try:
            sub_res = subprocess.run(["qsub", str(pbs_script)], capture_output=True, text=True, check=True)
            pbs_id = sub_res.stdout.strip()
            
            logger.info(f"Submitted PBS Job: {safe_filename} (Procedure: {procedure}, PBS ID: {pbs_id})")
            self.running_jobs[job_id] = {
                "procedure": procedure,
                "run_condition": run_condition,
                "out_file": output_file,
                "pbs_id": pbs_id
            }
            
            self._save_active_jobs_state()
            
            return True
        except Exception as e:
            logger.error(f"Failed to submit {job_id}: {e}")
            return False

    def _collect_results(self) -> None:
        completed_ids = []
        
        active_pbs_jobs = self._get_active_pbs_jobs()
        
        for job_id, info in self.running_jobs.items():
            out_file = info["out_file"]
            pbs_id = info.get("pbs_id", "")
            
            # Check if this specific job is still alive in qstat
            is_active = pbs_id in active_pbs_jobs or pbs_id.split('.')[0] in active_pbs_jobs
            
            # If the file doesn't exist, PBS is still running it
            if not out_file.exists():
                if is_active:
                    # File is missing, but job is still queued/running. Wait patiently.
                    continue
                else:
                    # File is missing AND job is gone from qstat = CRASH
                    logger.warning(f"Job {job_id} (PBS {pbs_id}) disappeared from qstat without output. Marking as crashed.")
                    result = JobResult(
                        job_id=job_id,
                        run_condition=info["run_condition"],
                        success=False,
                        error=f"PBS Job {pbs_id} crashed/killed (Not in qstat, no output generated)",
                        critical_point=None,
                        execution_time=0.0,
                        add_edge=False
                    )
                    completed_ids.append(job_id)
            else:
                # File exists! It finished normally (success or graceful failure)
                completed_ids.append(job_id)
                rc = info["run_condition"]
                
                try:
                    with open(out_file, 'r') as f:
                        res_data = json.load(f)
                        
                    cp = CriticalPoint.from_config(res_data["critical_point"]) if res_data.get("critical_point") else None
                    
                    result = JobResult(
                        job_id=job_id,
                        run_condition=rc,
                        success=res_data.get("success", False),
                        error=res_data.get("error"),
                        critical_point=cp,
                        execution_time=res_data.get("execution_time", 0.0),
                        add_edge=res_data.get("add_edge", False)
                    )
                except Exception as e:
                    result = JobResult(job_id=job_id,
                                       run_condition=rc,
                                       success=False,
                                       error=f"Output parsing error: {e}",
                                       critical_point=None,
                                       execution_time=0.0,
                                       add_edge=False)
                
            self.monitor.record_job_completion(
                procedure=info["procedure"],
                job_id=job_id,
                success=result.success,
                execution_time=result.execution_time,
                node_id=(info["run_condition"].node_id if hasattr(info["run_condition"], "node_id") else job_id),
            )
            
            self._handle_job_result(result, info["procedure"])
            
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_state_change(self)

        # remove finished jobs from tracking dictionary
        if completed_ids:
            for jid in completed_ids:
                del self.running_jobs[jid]
            
            # update the state file now that jobs are cleared out
            self._save_active_jobs_state()

    def _save_active_jobs_state(self) -> None:
        """Continuously dumps active jobs to a JSON file to survive master script crashes."""
        state_file = self.job_dir / "active_jobs.json"
        state = {}
        for jid, info in self.running_jobs.items():
            state[jid] = {
                "procedure": info["procedure"],
                "run_condition": info["run_condition"].to_config(),
                "out_file": str(info["out_file"]),
                "pbs_id": info["pbs_id"]
            }
            
        # Write to a temporary file first, then replace, to ensure atomic writes. 
        # This prevents file corruption if the script is killed mid-write.
        temp_file = state_file.with_suffix('.tmp')
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        temp_file.replace(state_file)

    def _recover_active_jobs(self) -> None:
        """Loads orphaned jobs and reconciles internal orchestrator trackers."""
        state_file = self.job_dir / "active_jobs.json"
        if not state_file.exists():
            return
            
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                
            for jid, info in state.items():
                procedure = info["procedure"]
                rc = RunCondition.from_config(info["run_condition"])
                
                # 1. Reconstruct the running_jobs dictionary
                self.running_jobs[jid] = {
                    "procedure": procedure,
                    "run_condition": rc,
                    "out_file": Path(info["out_file"]),
                    "pbs_id": info["pbs_id"]
                }
                
                # 2. Reconcile Queues: If the checkpoint was saved BEFORE the job was submitted, 
                # the job will still be in the queue. We must remove it.
                try:
                    self.job_queues.remove_job(procedure, jid)
                except Exception:
                    pass # It wasn't in the queue, which is fine.
                
                # 3. Patch Trackers: Make sure the orchestrator knows these nodes are "processing"
                node_id = self._get_parent_node_id(rc, jid)
                self.job_tracker.add_jobs_for_node(node_id, [jid])
                self.node_states.mark_processing(node_id, procedure)
                
            logger.info(f"Successfully recovered {len(self.running_jobs)} active PBS jobs from state file.")
            
        except Exception as e:
            logger.error(f"Failed to recover active jobs state: {e}")