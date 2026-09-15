from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

from .search_primitives import RunCondition
from .point import CriticalPoint

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    job_id: str
    run_condition: RunCondition
    critical_point: Optional[CriticalPoint] = None
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    add_edge: bool = False

    def __str__(self) -> str:
        lines = [
            f"JobResult (ID: {self.job_id}):",
            f"  Success: {self.success}",
            f"  Execution time: {self.execution_time:.3f}s",
            f"  Add edge: {self.add_edge}"
        ]

        if self.error:
            lines.append(f"  Error: {self.error}")

        if self.critical_point:
            lines.append(
                f"  Critical point found: {self.critical_point.index}-th order"
            )
            lines.append(f"  Final energy: {self.critical_point.energy:.6f}")
        else:
            lines.append("  Critical point: None (run failed)")

        lines.append("  Run condition:")
        # Indent the run condition string
        run_condition_str = str(self.run_condition)
        indented_rc = "\n".join("    " + line for line in run_condition_str.split("\n"))
        lines.append(indented_rc)

        return "\n".join(lines)
