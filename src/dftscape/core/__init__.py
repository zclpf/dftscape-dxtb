"""Core DFTScape components for molecular dynamics and critical point finding.

This module provides the fundamental building blocks for DFTScape's molecular
dynamics simulations and critical point location algorithms. It integrates
specialized components for efficient navigation of potential energy surfaces.

Core Components
---------------
Dynamics Framework:
- Dynamics: Abstract base class defining the interface for all dynamics simulations
- HiSD: High index Saddle Dynamics for finding transition states

Computational Backends:
- Backend: Abstract interface for quantum chemistry computations
- Backend implementations for different quantum chemistry packages

Eigensolvers:
- Multiple eigensolver implementations for Hessian eigenvalue computations
- Optimized for large-scale molecular systems

Symmetry and Constraints:
- SymmetryHandler: Manages molecular symmetry constraints
- SubspaceHandler: Handles subspace projections and constraints

Search and Orchestration:
- SearchOrchestrator: Coordinates complex search workflows
- JobExecutor: Manages parallel computation jobs
- NodeStateManager: Tracks computational node states

Key Concepts
------------
Critical Points: Stationary points on the potential energy surface where the gradient is zero
- Minima: Local energy minima (all Hessian eigenvalues positive)
- Saddle Points: Transition states (one negative Hessian eigenvalue)

Dynamics Algorithms:
- Use gradient information to guide molecular motion
- Employ Hessian information for improved convergence
- Handle symmetry constraints and subspace projections
- Monitor convergence and detect numerical instabilities

Usage Examples
--------------
Basic usage with HiSD for saddle point search:

    >>> from dftscape.core import HiSD, Backend, SubspaceHandler, SymmetryHandler
    >>> from dftscape.backends import BackendRegistry
    >>>
    >>> # Initialize components
    >>> backend = BackendRegistry.create('psi4', method='hf/sto-3g')
    >>> subspace_handler = SubspaceHandler(backend)
    >>> symmetry_handler = SymmetryHandler()
    >>>
    >>> # Create HiSD dynamics
    >>> hisd = HiSD(backend, subspace_handler, symmetry_handler)
    >>>
    >>> # Run simulation
    >>> result = hisd.run(run_condition)

Advanced configuration:

    >>> hisd = HiSD(
    ...     backend=backend,
    ...     subspace_handler=subspace_handler,
    ...     symmetry_handler=symmetry_handler,
    ...     dt=0.3,                    # Smaller time step
    ...     maxiter=5000,              # More iterations
    ...     grad_norm_tol=1e-7,        # Tighter convergence
    ...     subspace_update_interval=5 # Update subspace every 5 steps
    ... )

Implementation Notes
-------------------
- All dynamics classes inherit from the abstract Dynamics base class
- HiSD uses subspace projections to separate stable/unstable modes
- Energy explosion detection prevents numerical instabilities
- Configurable logging and metrics for monitoring progress
- Support for symmetry constraints in molecular systems
"""

from .interfaces import Backend

from .eigensolvers import (
    Eigensolver,
    ScipyLOBPCGEigensolver,
    EigshEigensolver,
    EighEigensolver,
    LOBPCGManualEigensolver,
    LOBPCGSimplifiedEigensolver,
)
from .dynamics import Dynamics, HiSD
from .search_primitives import RunCondition, RunConditionBuilder
from .symmetry_handlers import (
    SymmetryHandler,
    NoSymmetryHandler,
    MolecularSymmetryHandler,
    SymmetryRegistry,
)
from .subspace_handlers import SubspaceHandler
from .metrics import Metric, EnergyMetric, GradNormMetric
from .point import CriticalPoint
from .graph import CriticalPointGraph, SearchableCriticalPointGraph
from .search_orchestrator import (
    SearchOrchestrator,
    JobTracker,
    SearchOrchestratorRegistry,
)
from .node_state_manager import NodeStateManager
from .job_executor import JobExecutor

__all__ = [
    "Backend",
    "SubspaceHandler",
    "Dynamics",
    "HiSD",
    "RunCondition",
    "RunConditionBuilder",
    "SymmetryHandler",
    "NoSymmetryHandler",
    "MolecularSymmetryHandler",
    "SymmetryRegistry",
    "ScipyLOBPCGEigensolver",
    "EigshEigensolver",
    "EighEigensolver",
    "Eigensolver",
    "LOBPCGManualEigensolver",
    "LOBPCGSimplifiedEigensolver",
    "Metric",
    "EnergyMetric",
    "GradNormMetric",
    "CriticalPoint",
    "CriticalPointGraph",
    "SearchableCriticalPointGraph",
    "SearchOrchestrator",
    "JobTracker",
    "NodeStateManager",
    "JobExecutor",
    "SearchOrchestratorRegistry",
]
