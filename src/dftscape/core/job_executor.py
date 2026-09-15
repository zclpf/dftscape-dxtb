from __future__ import annotations
from typing import List
import time
import logging

from .search_primitives import RunCondition
from .job_result import JobResult
from .interfaces import HiSDRunner

# Set up logging
logger = logging.getLogger(__name__)


# TODO: initialise backend here for thread/parallelisation safety


class JobExecutor:

    def __init__(self, hisd_runner: HiSDRunner, max_concurrent: int = 1):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        self.hisd_runner = hisd_runner
        self.max_concurrent = max_concurrent

    def execute_job(self, run_condition: RunCondition, job_id: str) -> JobResult:
        start_time = time.time()

        try:
            critical_point, add_edge = self.hisd_runner.run(run_condition)
            execution_time = time.time() - start_time

            if critical_point is None:
                # HiSD failed to converge within maxiter
                result = JobResult(
                    job_id=job_id,
                    run_condition=run_condition,
                    critical_point=None,
                    success=False,
                    error="HiSD failed to converge.",
                    execution_time=execution_time,
                    add_edge=False
                )
                return result

            result = JobResult(
                job_id=job_id,
                run_condition=run_condition,
                critical_point=critical_point,
                success=True,
                execution_time=execution_time,
                add_edge=add_edge
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Job execution failed: {str(e)}"

            result = JobResult(
                job_id=job_id,
                run_condition=run_condition,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                add_edge=False
            )

            logger.error(f"Job {job_id} failed: {error_msg}")
            return result

    def execute_batch(self, jobs: List[tuple[str, RunCondition]]) -> List[JobResult]:
        results = []
        for job_id, run_condition in jobs:
            result = self.execute_job(run_condition, job_id)
            results.append(result)
        return results
