"""
DFTScape: Critical Point Landscape Exploration
=======================================================

DFTScape is a specialized framework for systematic exploration of molecular
potential energy landscapes through advanced search algorithms. It focuses on
discovering and characterizing critical points (minima, saddles, maxima) and
their connectivity using High index saddle dynamics (HiSD) methods.

Key Features
============

Critical Point Discovery:
    - Systematic exploration of potential energy landscapes
    - Discovery of minima, saddles, and maxima using HiSD methods
    - Characterization of critical point properties (energies, Hessians)
    - Automatic classification of stationary point types

Advanced Search Algorithms:
    - High index saddle dynamics (HiSD) for critical point finding
    - Upward and downward search directions for comprehensive landscape mapping
    - Parallel search execution with intelligent job queue management
    - Adaptive search strategies based on landscape topology

Landscape Connectivity Analysis:
    - Construction of searchable critical point graphs
    - Identification of reaction pathways and transition networks
    - Analysis of energy barriers and reaction mechanisms
    - Visualization of connected critical point networks

Multiple Computational Backends:
    - Psi4: Production quantum chemistry calculations
    - Test: Analytical test functions for algorithm validation
    - Extensible: Easy integration of custom backends (ML-IAP, force fields, etc.)

Robust Execution Framework:
    - Parallel job execution with dependency management
    - Comprehensive checkpointing and recovery capabilities
    - Real-time progress monitoring and analytics
    - Error handling and job failure recovery

Installation
============

*pip installer instruction will be updated after publishing*

## Manual installation

Clone the repository

```shell
git clone https://github.com/MLDS-NUS/dftscape.git
cd dftscape
```

### 📦 Step 1: Install uv and set up the environment

Packages are managed with [`uv`](https://docs.astral.sh/uv/).

**Install `uv`** 🛠️
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Set up the environment** 🌍
```shell
uv python install 3.13
uv sync
uv pip install -e .
```

### 🔬 Step 2: Install the Psi4 Backend

*🔮 More backend support coming in the future!*

The following is an example for **MacOS Silicon** with `python 3.13`.
Find the release suitable for your system [here](https://psicode.org/installs/v191/) and substitute accordingly.

**Download binary package** 📥
```shell
curl "https://vergil.chemistry.gatech.edu/psicode-download/Psi4conda-1.9.1-py313-MacOSX-arm64.sh" -o Psi4conda-1.9.1-py313-MacOSX-arm64.sh --keepalive-time 2
```

**Install the `Psi4` package** ⚙️
(Assume location is `$HOME/psi4-1.9.1`)
```shell
INSTALL="$HOME/psi4-1.9.1"
bash Psi4conda-1.9.1-py313-MacOSX-arm64.sh -b -p "$INSTALL"
```

You should now have something like this (check with `ls "$INSTALL"`)
```shell
$INSTALL/
  bin/
  lib/
  pkgs/
  ...
  lib/python3.11/site-packages/psi4/  # <-- the Python module we want to import
```

**Activate local environment** 🔑
```shell
source .venv/bin/activate
python -V  # <-- verify the version (e.g., 3.13.x)
```

**Make venv see Psi4** 🔗
```shell
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # <-- find this venv's site-packages
echo "$INSTALL/lib/python3.13/site-packages" > "$SP/psi4_external.pth"  # <-- write a .pth that adds the installer site-packages to sys.path
```

**Verify installation** ✅
```shell
python - <<'PY'
import sys, psi4
print("Python:", sys.version.split()[0])
print("Psi4:", psi4.__version__)
print("Psi4 module from:", psi4.__file__)
PY
```

Quick Start
===========

Basic Landscape Exploration:
    >>> from dftscape.backends import BackendRegistry
    >>> from dftscape.core.search_orchestrator import SearchOrchestrator
    >>> from dftscape.core.graph import SearchableCriticalPointGraph
    >>> from dftscape.core.dynamics import HiSD
    >>> from dftscape.core.point import CriticalPoint
    >>> from dftscape.common.checkpoint import JSONCheckpointer
    >>> import numpy as np
    >>>
    >>> # Set up quantum chemistry backend
    >>> backend = BackendRegistry.create("psi4",
    ...     cmp_mthd="b3lyp/6-31g*",
    ...     memory="2GB",
    ...     num_threads=4,
    ...     molecule_xyz="3\\n\\nO 0.0 0.0 0.0\\nH 0.757 0.586 0.0\\nH -0.757 0.586 0.0",
    ...     molecule_directives={"charge": 0, "multiplicity": 1}
    ... )
    >>>
    >>> # Initialize critical point graph
    >>> graph = SearchableCriticalPointGraph()
    >>>
    >>> # Set up HiSD dynamics
    >>> hisd = HiSD(backend=backend, dt=0.1, ...)
    >>>
    >>> # Create search orchestrator
    >>> orchestrator = SearchOrchestrator(
    ...     graph=graph,
    ...     hisd_runner=hisd,
    ...     d_max=3,  # Maximum search order
    ...     perturbation_magnitude=0.1
    ... )
    >>>
    >>> # Load initial critical point (e.g., global minimum)
    >>> checkpointer = JSONCheckpointer()
    >>> initial_point_data = checkpointer.load("minimum.json")
    >>> initial_point = CriticalPoint.from_config(initial_point_data)
    >>> graph.add(initial_point)
    >>>
    >>> # Run upward search to find saddles and maxima
    >>> orchestrator.run_upward_search()
    >>>
    >>> # Analyze discovered critical points
    >>> print(f"Found {len(graph.nodes)} critical points")
    >>> for node_id, point in graph.nodes.items():
    ...     print(f"Point {node_id}: Energy = {point.energy:.6f}, Index = {point.index}")

Critical Point Characterization:
    >>> from dftscape.core.point import CriticalPoint
    >>>
    >>> # Create critical point from coordinates
    >>> coords = np.array([0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0])
    >>> point = CriticalPoint.from_state(coords, backend)
    >>>
    >>> # Analyze critical point properties
    >>> eigenvals = np.linalg.eigvals(point.hessian)
    >>> n_negative = np.sum(eigenvals < 0)
    >>> point.index = n_negative
    >>>
    >>> if n_negative == 0:
    ...     print("Minimum found")
    >>> elif n_negative == 1:
    ...     print("Transition state found")
    >>> else:
    ...     print("Higher-order saddle found")

Architecture Overview
=====================

DFTScape follows a modular architecture designed for systematic landscape exploration:

Core Components:
    - backends: Computational engines for energy/derivative calculations
    - core: Search algorithms, graph management, and execution coordination
    - checkpoint: Data persistence and recovery for long-running searches

Search System:
    The search system uses HiSD (High index saddle dynamics) to
    systematically explore potential energy landscapes. It can search in both
    upward and downward directions to discover all critical points connected
    to a given starting point.

Graph Management:
    Critical points are organized in a searchable graph structure that
    maintains connectivity information and enables efficient querying of
    reaction pathways and energy landscapes.

Execution Framework:
    The orchestrator manages parallel job execution, dependency tracking,
    and resource allocation for efficient landscape exploration.

Main Modules
============

dftscape.backends:
    Computational backend implementations and registry system.

    Available backends:
    - Psi4Backend: Production quantum chemistry with Psi4
    - TestBackend: Analytical test functions for validation

dftscape.core:
    Core search functionality and landscape exploration algorithms.

    Key components:
    - SearchOrchestrator: Main coordination system for search execution
    - HiSD: High index saddle dynamics for critical point finding
    - SearchableCriticalPointGraph: Graph structure for critical points
    - CriticalPoint: Data structure for stationary point characterization

dftscape.checkpoint:
    Data persistence and recovery system for long-running searches.

    Features:
    - Multiple storage formats (NPZ, JSON)
    - Automatic checkpointing during search
    - Recovery from interruptions
    - Export of critical point networks

Implementing Custom Backends
=============================

DFTScape supports custom backend implementations for specialized computational
methods. This enables integration of machine learning interatomic potentials,
force fields, or any computational method that can provide energies and derivatives.

Backend Protocol Requirements:
    All backends must implement the Backend protocol with four methods:
    - compute_energy(state): Calculate potential energy
    - compute_gradient(state): Calculate nuclear gradients
    - compute_hessian(state): Calculate nuclear Hessian
    - get_xyz(state): Export coordinates in XYZ format

For detailed implementation guidance, see:
    dftscape.backends package documentation


Example:
    >>> from dftscape.backends import BackendRegistry
    >>> from dftscape.core.interfaces import Backend
    >>>
    >>> @BackendRegistry.register("ml_iap")
    >>> class MLInteratomicPotentialBackend(Backend):
    ...     def __init__(self, model_path: str, atomic_numbers: list[int]):
    ...         self.model = load_ml_model(model_path)
    ...         self.atomic_numbers = atomic_numbers
    ...
    ...     def compute_energy(self, state):
    ...         coords = state.reshape(-1, 3)
    ...         return self.model.predict_energy(coords, self.atomic_numbers)
    ...
    ...     def compute_gradient(self, state):
    ...         coords = state.reshape(-1, 3)
    ...         forces = self.model.predict_forces(coords, self.atomic_numbers)
    ...         return -forces.flatten()  # Convert forces to gradients
    ...
    ...     def compute_hessian(self, state):
    ...         # Use finite differences or analytical Hessian if available
    ...         return self._finite_difference_hessian(state)
    ...
    ...     def get_xyz(self, state):
    ...         coords = state.reshape(-1, 3)
    ...         return generate_xyz_string(coords, self.atomic_numbers)

The custom backend automatically integrates with DFTScape search algorithms:
    >>> from dftscape.backends import BackendRegistry
    >>> from dftscape.core.search_orchestrator import SearchOrchestrator
    >>> from dftscape.core.graph import SearchableCriticalPointGraph
    >>> from dftscape.core.dynamics import HiSD
    >>> from dftscape.core.point import load_critical_point_from_file  # Assuming this exists
    >>>
    >>> # Initialize critical point graph (added for completeness)
    >>> graph = SearchableCriticalPointGraph()
    >>>
    >>> # Load initial critical point (added for completeness)
    >>> initial_point = load_critical_point_from_file("minimum.json")
    >>> graph.add_node(initial_point)
    >>>
    >>> # Use custom backend in landscape exploration
    >>> hisd = HiSD(backend=BackendRegistry.create("ml_iap", model_path="model.pth"))
    >>> orchestrator = SearchOrchestrator(graph=graph, hisd_runner=hisd)
    >>> orchestrator.run_upward_search()

For comprehensive implementation examples and best practices, refer to:
    dftscape.backends package documentation

API Reference
=============

Core Classes:
    BackendRegistry: Registry for computational backends
    SearchOrchestrator: Main coordination system for landscape exploration
    HiSD: High index saddle dynamics
    SearchableCriticalPointGraph: Graph structure for critical points
    CriticalPoint: Data structure for stationary point characterization

Backend Protocol:
    Backend: Protocol defining computational backend interface
    Psi4Backend: Psi4 quantum chemistry backend
    TestBackend: Analytical test function backend

Search Components:
    SearchStrategy: Base class for search algorithm implementations
    JobTracker: Job lifecycle and dependency management
    NodeStateManager: Processing state coordination
    SearchMonitor: Performance tracking and analytics

Data Structures:
    CriticalPoint: Stationary point with coordinates, energy, gradient, Hessian
    SearchResult: Search operation result container
    CheckpointData: Checkpoint data structure for persistence

Contributing and Extending
==========================

DFTScape is designed for easy extension and contribution:

Adding New Backends:
    1. Implement the Backend protocol
    2. Register with @BackendRegistry.register decorator
    3. Add comprehensive docstrings and examples
    4. Include tests and validation

Adding New Search Strategies:
    1. Extend SearchStrategy base class
    2. Implement search logic for critical point discovery
    3. Add to search orchestrator with proper configuration
    4. Provide usage examples and benchmarks

Code Quality:
    - Follow PEP 8 style guidelines
    - Include comprehensive docstrings
    - Add unit tests for new functionality
    - Update documentation for API changes

Testing:
    - Use pytest for unit testing
    - Include integration tests for search algorithms
    - Test against known analytical systems
    - Validate critical point characterization

Documentation:
    - Update docstrings for all public APIs
    - Add examples to documentation
    - Include mathematical formulations where relevant
    - Document limitations and assumptions

See Also
========

- dftscape.backends: Backend implementations and custom backend development
- dftscape.core: Core search algorithms and landscape exploration
- dftscape.checkpoint: Data persistence and recovery
- Documentation: https://dftscape.readthedocs.io/
- Repository: https://github.com/MLDS-NUS/dftscape
"""

# Package version
try:
    from importlib.metadata import version

    __version__ = version("dftscape")
except Exception:
    __version__ = "0.0.0"
