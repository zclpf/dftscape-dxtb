"""
Test Backend Implementation for DFTScape
========================================

This module provides a test backend implementation for algorithm testing and
validation in DFTScape. It implements a 3D separable double-well potential
using JAX for automatic differentiation, making it ideal for testing search
algorithms and optimization methods.

The test backend serves as a controlled environment for validating DFTScape's
search algorithms with well-characterized stationary points and known
analytical properties.

Key Features
============

Analytical Test Function:
    Implements the separable double-well potential:
    f(x, y, z) = (x²-1)² + (y²-1)² + (z²-1)²

    This function has well-characterized critical points with known Morse indices.

JAX Integration:
    Uses JAX for automatic differentiation and JIT compilation
    Provides exact analytical gradients and Hessians
    Optional dependency - backend gracefully degrades if JAX unavailable

Well-Characterized Landscape:
    - 8 global minima at (±1, ±1, ±1) with f = 0
    - 12 index-1 saddles with f = 1
    - 6 index-2 saddles with f = 2
    - 1 index-3 saddle (maximum) at (0,0,0) with f = 3

Performance Optimized:
    JIT-compiled functions for fast evaluation
    Efficient gradient and Hessian computation
    Minimal memory overhead

Critical Points Analysis
========================

The potential function has 27 critical points with the following distribution:

Global Minima (Index 0):
    Points: (±1, ±1, ±1) - all combinations
    Energy: f = 0
    Count: 8 points

Index-1 Saddles:
    Points: One coordinate = 0, others = ±1
    Energy: f = 1
    Count: 12 points

Index-2 Saddles:
    Points: Two coordinates = 0, one = ±1
    Energy: f = 2
    Count: 6 points

Index-3 Saddle (Maximum):
    Point: (0, 0, 0)
    Energy: f = 3
    Count: 1 point

All critical points are non-degenerate with Hessian eigenvalues {-4, 8}.

Usage Examples
==============

Basic Usage:
    >>> from dftscape.backends import BackendRegistry
    >>> # backend = BackendRegistry.create("test")
    >>> # state = np.array([0.5, 0.5, 0.5])
    >>> # energy = backend.compute_energy(state)
    >>> # gradient = backend.compute_gradient(state)
    >>> # hessian = backend.compute_hessian(state)

Testing Optimization:
    >>> # Start near a saddle point
    >>> # initial_state = np.array([0.0, 1.0, 1.0])  # Index-1 saddle
    >>> # energy = backend.compute_energy(initial_state)
    >>> # print(f"Energy at saddle: {energy}")  # Should be ~1.0

XYZ Format Output:
    >>> # xyz_string = backend.get_xyz(state)
    >>> # print(xyz_string)
    >>> # 1
    >>> # Test molecule
    >>> # X     0.500000     0.500000     0.500000

Dependencies
============

JAX (Optional):
    Required for automatic differentiation and JIT compilation
    Install with: pip install jax jaxlib
    If unavailable, backend registration is skipped with informative error

Notes
=====

- Designed specifically for testing and validation of search algorithms
- All derivatives computed analytically using automatic differentiation
- Provides exact gradients and Hessians (no numerical approximations)
- 3D coordinate system with separable potential for easy analysis
- Useful for benchmarking optimization algorithms and transition state finding
- Critical points have known analytical properties for validation

See Also
========

- dftscape.backends.registry: Backend registry and protocol definitions
- dftscape.backends.impl_psi4: Production Psi4 quantum chemistry backend
"""

from __future__ import annotations
from typing import Any
import numpy as np

from ..core.interfaces import Backend
from .registry import BackendRegistry

# Try to import JAX - if not available, this backend won't be registered
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian, jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy objects to avoid import errors
    jnp = None
    jax = None
    grad = None
    hessian = None
    jit = None

# Try to import JAX - if not available, this backend won't be registered
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian, jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy objects to avoid import errors
    jnp = None
    jax = None
    grad = None
    hessian = None
    jit = None


if JAX_AVAILABLE:

    @BackendRegistry.register("test")
    class TestBackend(Backend):
        """
        Test Backend for Algorithm Validation and Testing.

        This backend implements a 3D separable double-well potential function
        designed specifically for testing and validating search algorithms in DFTScape.
        The potential function provides a controlled environment with well-characterized
        stationary points, making it ideal for benchmarking optimization methods and
        transition state finding algorithms.

        Mathematical Function:
            f(x, y, z) = (x²-1)² + (y²-1)² + (z²-1)²

        The function is separable in each coordinate, allowing for easy analysis
        of the optimization landscape and critical point properties.

        Critical Points (27 total):
        ===========================

        Global Minima (Index 0):
            Locations: All combinations of (±1, ±1, ±1)
            Energy: f = 0
            Count: 8 points
            Description: Eight equivalent global minima at the corners of the cube

        Index-1 Saddles:
            Locations: One coordinate = 0, others = ±1 (all combinations)
            Energy: f = 1
            Count: 12 points
            Description: Saddle points on the faces of the potential energy cube

        Index-2 Saddles:
            Locations: Two coordinates = 0, one = ±1 (all combinations)
            Energy: f = 2
            Count: 6 points
            Description: Saddle points along the edges of the potential energy cube

        Index-3 Saddle (Maximum):
            Location: (0, 0, 0)
            Energy: f = 3
            Count: 1 point
            Description: Global maximum at the center of the coordinate system

        All critical points are non-degenerate with Hessian eigenvalues {-4, 8},
        providing well-conditioned test cases for optimization algorithms.

        Attributes:
            config (Dict[str, Any]): Configuration parameters passed during initialization
            _potential_fn (Callable): JIT-compiled potential energy function
            _gradient_fn (Callable): JIT-compiled gradient function
            _hessian_fn (Callable): JIT-compiled Hessian function

        Notes:
            - Requires JAX for automatic differentiation and JIT compilation
            - All derivatives computed analytically (no numerical approximations)
            - Functions are JIT-compiled for optimal performance
            - Thread-safe for evaluation (no internal state modifications)
            - Designed for 3D coordinate systems only

        Examples:
            >>> backend = TestBackend()
            >>> # Evaluate at a minimum
            >>> min_point = np.array([1.0, 1.0, 1.0])
            >>> energy = backend.compute_energy(min_point)
            >>> print(f"Energy at minimum: {energy}")  # 0.0
            >>>
            >>> # Evaluate at the maximum
            >>> max_point = np.array([0.0, 0.0, 0.0])
            >>> energy = backend.compute_energy(max_point)
            >>> print(f"Energy at maximum: {energy}")  # 3.0
        """

        def __init__(self, **kwargs: Any):
            """
            Initialize the JAX test backend with configuration parameters.

            Sets up the 3D separable double-well potential function with JAX
            automatic differentiation and JIT compilation for optimal performance.
            The backend is configured with the mathematical function:

            f(x, y, z) = (x²-1)² + (y²-1)² + (z²-1)²

            Args:
                **kwargs: Configuration parameters (stored but not currently used).
                         Reserved for future extensions and customization options.

            Raises:
                ImportError: If JAX is not available (handled at class level).

            Notes:
                - Creates JIT-compiled versions of potential, gradient, and Hessian
                - Functions are optimized for repeated evaluations
                - No internal state is modified during evaluations (thread-safe)
                - Configuration parameters are stored for future extensibility

            Examples:
                >>> backend = TestBackend(precision="high", cache=True)
                >>> # Configuration stored but not currently used
            """
            super().__init__()
            # Store any configuration parameters
            self.config = kwargs

            # Define the separable double-well potential function using JAX
            def potential_function(state: Any) -> float:
                x, y, z = state[0], state[1], state[2]
                return (
                    (x**2 - 1) ** 2
                    + (y**2 - 1) ** 2
                    + (z**2 - 1) ** 2
                    # + 0.1 * (2 * x + y + 0.3 * z)
                )

            # Create JIT-compiled versions for better performance
            self._potential_fn = jit(potential_function)
            self._gradient_fn = jit(grad(potential_function))
            self._hessian_fn = jit(hessian(potential_function))

        def compute_energy(self, state: np.ndarray) -> float:
            """
            Compute the potential energy at the given 3D state.

            Evaluates the separable double-well potential function:
            f(x, y, z) = (x²-1)² + (y²-1)² + (z²-1)²

            This function has known analytical properties and is designed for
            testing optimization algorithms with well-characterized critical points.

            Args:
                state: 1D numpy array with shape (3,) containing [x, y, z] coordinates.
                      Must be exactly 3-dimensional for this test backend.

            Returns:
                float: Potential energy value at the given coordinates.
                      Range: [0, ∞), with global minimum at 0 and no upper bound.

            Raises:
                ValueError: If state does not have exactly 3 dimensions.

            Examples:
                >>> backend = TestBackend()
                >>> # Global minimum
                >>> min_state = np.array([1.0, 1.0, 1.0])
                >>> energy = backend.compute_energy(min_state)
                >>> print(f"Global minimum energy: {energy}")  # 0.0
                >>>
                >>> # Index-1 saddle point
                >>> saddle_state = np.array([0.0, 1.0, 1.0])
                >>> energy = backend.compute_energy(saddle_state)
                >>> print(f"Saddle energy: {energy}")  # 1.0
                >>>
                >>> # Global maximum
                >>> max_state = np.array([0.0, 0.0, 0.0])
                >>> energy = backend.compute_energy(max_state)
                >>> print(f"Global maximum energy: {energy}")  # 3.0

            Notes:
                - Uses JIT-compiled JAX function for optimal performance
                - Computation is deterministic and thread-safe
                - No side effects or internal state modifications
                - Energy landscape is symmetric in all three coordinates
            """
            if len(state) != 3:
                raise ValueError(
                    f"TestBackend expects 3D state, got {len(state)} dimensions"
                )

            # Convert numpy array to JAX array
            state_jax = jnp.array(state)
            energy = self._potential_fn(state_jax)

            # Convert back to float for compatibility
            return float(energy)

        def compute_gradient(self, state: np.ndarray) -> np.ndarray:
            """
            Compute the analytical gradient vector using automatic differentiation.

            Calculates the exact gradient of the separable double-well potential:
            ∇f(x, y, z) = [2(x²-1)*2x, 2(y²-1)*2y, 2(z²-1)*2z]

            The gradient is computed using JAX's automatic differentiation,
            providing exact analytical derivatives without numerical approximations.

            Args:
                state: 1D numpy array with shape (3,) containing [x, y, z] coordinates.
                      Must be exactly 3-dimensional for this test backend.

            Returns:
                np.ndarray: Gradient vector with shape (3,) containing [df/dx, df/dy, df/dz].
                           Each component is the partial derivative with respect to that coordinate.

            Raises:
                ValueError: If state does not have exactly 3 dimensions.

            Examples:
                >>> backend = TestBackend()
                >>> # Gradient at global minimum (should be zero)
                >>> min_state = np.array([1.0, 1.0, 1.0])
                >>> grad = backend.compute_gradient(min_state)
                >>> print(f"Gradient at minimum: {grad}")  # [0. 0. 0.]
                >>>
                >>> # Gradient at origin (global maximum)
                >>> origin = np.array([0.0, 0.0, 0.0])
                >>> grad = backend.compute_gradient(origin)
                >>> print(f"Gradient at origin: {grad}")  # [0. 0. 0.]
                >>>
                >>> # Gradient at a point away from critical points
                >>> point = np.array([0.5, 0.5, 0.5])
                >>> grad = backend.compute_gradient(point)
                >>> print(f"Gradient at (0.5,0.5,0.5): {grad}")

            Notes:
                - Uses JIT-compiled JAX automatic differentiation
                - Provides exact analytical gradients (no numerical errors)
                - Computation is deterministic and thread-safe
                - Gradient vanishes at all critical points (minima, saddles, maximum)
                - Due to separability, each coordinate's gradient depends only on itself
            """
            if len(state) != 3:
                raise ValueError(
                    f"TestBackend expects 3D state, got {len(state)} dimensions"
                )

            # Convert numpy array to JAX array
            state_jax = jnp.array(state)
            gradient_jax = self._gradient_fn(state_jax)

            # Convert back to numpy array for compatibility
            return np.array(gradient_jax)

        def compute_hessian(self, state: np.ndarray) -> np.ndarray:
            """
            Compute the analytical Hessian matrix using automatic differentiation.

            Calculates the exact Hessian matrix of the separable double-well potential.
            Due to the separable nature of the function, the Hessian is diagonal:

            H(x,y,z) = diag[4(3x²-1), 4(3y²-1), 4(3z²-1)]

            The Hessian is computed using JAX's automatic differentiation of the gradient,
            providing exact analytical second derivatives.

            Args:
                state: 1D numpy array with shape (3,) containing [x, y, z] coordinates.
                      Must be exactly 3-dimensional for this test backend.

            Returns:
                np.ndarray: 3x3 Hessian matrix containing second derivatives.
                           Matrix is symmetric and diagonal due to function separability.

            Raises:
                ValueError: If state does not have exactly 3 dimensions.

            Examples:
                >>> backend = TestBackend()
                >>> # Hessian at global minimum
                >>> min_state = np.array([1.0, 1.0, 1.0])
                >>> hess = backend.compute_hessian(min_state)
                >>> print("Hessian at minimum:")
                >>> print(hess)  # Diagonal matrix with positive eigenvalues
                >>>
                >>> # Hessian at origin (global maximum)
                >>> origin = np.array([0.0, 0.0, 0.0])
                >>> hess = backend.compute_hessian(origin)
                >>> print("Hessian at origin:")
                >>> print(hess)  # Diagonal matrix with negative eigenvalues
                >>>
                >>> # Check eigenvalues (should be [-4, -4, -4] at origin)
                >>> eigenvals = np.linalg.eigvals(hess)
                >>> print(f"Eigenvalues at origin: {eigenvals}")

            Notes:
                - Uses JIT-compiled JAX automatic differentiation
                - Provides exact analytical Hessian (no numerical approximations)
                - Hessian is always diagonal due to function separability
                - Eigenvalues determine Morse index of critical points:
                  * Positive eigenvalues: minima along those directions
                  * Negative eigenvalues: maxima along those directions
                - All critical points have non-degenerate Hessians
            """
            if len(state) != 3:
                raise ValueError(
                    f"TestBackend expects 3D state, got {len(state)} dimensions"
                )

            # Convert numpy array to JAX array
            state_jax = jnp.array(state)
            hessian_jax = self._hessian_fn(state_jax)

            # Convert back to numpy array for compatibility
            return np.array(hessian_jax)

        def get_xyz(self, state: np.ndarray) -> str:
            """
            Generate XYZ format string representation of the 3D state.

            For the test backend, the 3D coordinates are represented as a single
            "X" atom at the specified (x, y, z) position. This provides a minimal
            molecular representation suitable for visualization and compatibility
            with molecular analysis tools.

            The XYZ format is a standard text-based molecular structure format
            with the structure:
            - Line 1: Number of atoms (1 for test backend)
            - Line 2: Comment/title line
            - Line 3+: Atomic symbol and coordinates (X x y z)

            Args:
                state: 1D numpy array with shape (3,) containing [x, y, z] coordinates.
                      Must be exactly 3-dimensional for this test backend.

            Returns:
                str: XYZ format string with proper formatting and newlines.
                    Contains exactly 3 lines representing a single atom.

            Raises:
                ValueError: If state does not have exactly 3 dimensions.

            Examples:
                >>> backend = TestBackend()
                >>> # Get XYZ at global minimum
                >>> min_state = np.array([1.0, 1.0, 1.0])
                >>> xyz = backend.get_xyz(min_state)
                >>> print(xyz)
                1
                Test molecule
                X     1.000000     1.000000     1.000000
                >>>
                >>> # Get XYZ at origin
                >>> origin = np.array([0.0, 0.0, 0.0])
                >>> xyz = backend.get_xyz(origin)
                >>> print(xyz)
                1
                Test molecule
                X     0.000000     0.000000     0.000000

            Notes:
                - Always represents the state as a single "X" atom
                - Coordinates are formatted with 6 decimal places
                - Suitable for molecular visualization tools
                - Compatible with standard XYZ file format specifications
                - Comment line identifies this as a test molecule
            """
            if len(state) != 3:
                raise ValueError(
                    f"TestBackend expects 3D state, got {len(state)} dimensions"
                )

            x, y, z = state
            xyz_string = f"1\nTest molecule\nX {x:12.6f} {y:12.6f} {z:12.6f}"
            return xyz_string

else:
    # JAX is not available - create a dummy class that raises an informative error
    class TestBackend:
        """
        Dummy TestBackend class when JAX is not available.

        This class serves as a placeholder when the JAX library is not installed.
        It provides informative error messages to guide users toward proper
        installation of the required dependencies.

        The TestBackend requires JAX for automatic differentiation and JIT
        compilation of the mathematical functions. Without JAX, the backend
        cannot provide the analytical gradients and Hessians needed for
        algorithm testing.

        Installation:
            To use the TestBackend, install JAX with:
            pip install jax jaxlib

            Or with conda:
            conda install jax jaxlib

        Notes:
            - This class is only instantiated when JAX is not available
            - Any attempt to use this backend will raise ImportError
            - Serves as a graceful degradation mechanism
            - Provides clear installation instructions to users
        """

        def __init__(self, **kwargs: Any):
            """
            Initialize dummy backend and raise informative error.

            Immediately raises an ImportError with installation instructions
            since JAX is required for the TestBackend functionality.

            Args:
                **kwargs: Ignored configuration parameters.

            Raises:
                ImportError: Always raised with installation instructions for JAX.
            """
            raise ImportError(
                "JAX is required for TestBackend but is not installed. "
                "Install JAX with: pip install jax jaxlib"
            )
