"""
Metrics for monitoring and analyzing dynamics simulations.

This module provides a comprehensive framework for computing and tracking various
observables during molecular dynamics simulations. Metrics enable real-time
monitoring of simulation progress, convergence detection, and analysis of
molecular properties.

## Overview

The metrics system is designed to be:
- **Extensible**: Easy to add new metric types through inheritance
- **Configurable**: Metrics can be specified via YAML configuration
- **Efficient**: Metrics are computed on-demand with minimal overhead
- **Serializable**: Metrics can be saved and restored from configuration

## Key Components

### Metric (Abstract Base Class)
The foundation for all metrics, providing:
- Common interface for metric computation
- Configuration serialization/deserialization
- Registry integration for dynamic instantiation
- Formatting support for logging output

### MetricRegistry
A registry system that enables:
- Dynamic metric creation from configuration files
- Factory pattern for metric instantiation
- Support for parameterized metrics
- YAML configuration parsing

### Built-in Metrics

#### EnergyMetric
Computes the potential energy of the current molecular state.

#### GradNormMetric
Computes the Euclidean norm of the gradient vector, useful for convergence monitoring.

#### BondLengthMetric
Monitors the distance between two specified atoms, useful for reaction coordinate analysis.

#### HessEvalMaxMetric & HessEvalMinMetric
Monitor the maximum and minimum Hessian eigenvalues, particularly useful for saddle dynamics.

## Usage Examples

### Basic Usage
```python
from dftscape.core.metrics import EnergyMetric, GradNormMetric

# Create metrics
energy_metric = EnergyMetric()
grad_metric = GradNormMetric()

# Use in dynamics simulation
# (metrics are automatically called during simulation logging)
```

### Configuration-based Creation
```python
from dftscape.core.metrics import MetricRegistry

# From YAML-like configuration
config = [
    "energy",
    "grad_norm",
    {"bond_length": {"indices": [0, 1]}},
    "hess_eval_max"
]

metrics = MetricRegistry.parse_metrics_config(config)
```

### Custom Metric Implementation
```python
from dftscape.core.metrics import Metric, MetricRegistry

@MetricRegistry.register("custom_metric")
class CustomMetric(Metric):
    def __init__(self, param: float = 1.0):
        super().__init__("Custom", ":8.4f")
        self.param = param

    def compute(self, dynamics, step: int, dt: float) -> float:
        # Implement custom computation
        return some_value
```

## Configuration Format

Metrics can be specified in YAML configuration files:

```yaml
metrics:
  - energy                    # Simple metric
  - grad_norm                 # Another simple metric
  - bond_length:              # Parameterized metric
      indices: [0, 1]         # Bond between atoms 0 and 1
  - hess_eval_max             # Hessian eigenvalue monitoring
```

## Integration with Dynamics

Metrics are automatically integrated with dynamics simulations:
- Computed at regular intervals during simulation
- Logged with formatted output
- Used for convergence monitoring
- Can trigger early termination conditions

## Performance Considerations

- **Lazy Evaluation**: Metrics are only computed when needed
- **Caching**: Avoid redundant computations where possible
- **Formatting**: Efficient string formatting for logging
- **Memory**: Minimal memory footprint for most metrics

## Extending the System

To add a new metric:
1. Inherit from the `Metric` base class
2. Implement the `compute()` method
3. Register with `@MetricRegistry.register("name")`
4. Optionally implement custom configuration handling

The registry system automatically handles serialization, deserialization, and
factory creation for all registered metrics.
"""

from __future__ import annotations
from typing import cast, Union

from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
from ..common import BaseRegistry


__all__ = [
    "Metric",
    "MetricRegistry",
]


class Metric(ABC):
    """
    Abstract base class for all metrics used in dynamics simulations.

    A metric represents a computable observable that can be monitored during
    molecular dynamics simulations. Metrics provide real-time feedback about
    simulation progress, convergence status, and molecular properties.

    ## Key Features

    - **Abstract Interface**: Subclasses must implement the `compute()` method
    - **Configuration Support**: Automatic serialization/deserialization
    - **Registry Integration**: Dynamic instantiation from configuration
    - **Formatting**: Configurable output formatting for logging

    ## Attributes

    - `name`: Human-readable name for the metric (used in logging)
    - `format_str`: Python format string for displaying metric values
    - `_registry_key`: Internal key used by the registry system

    ## Usage

    ```python
    class CustomMetric(Metric):
        def __init__(self):
            super().__init__("Custom", ":8.4f")

        def compute(self, dynamics, step: int, dt: float) -> float:
            return some_computation(dynamics.state)
    ```

    ## Configuration

    Metrics can be configured via dictionaries for serialization:

    ```python
    config = metric.to_config()  # Serialize
    restored = Metric.from_config(config)  # Deserialize
    ```
    """

    def __init__(self, name: str, format_str: str = ":12.6f"):
        """
        Initialize a metric with name and formatting.

        Args:
            name: Human-readable name for the metric
            format_str: Python format string (default: ":12.6f")
        """
        self.name = name
        self.format_str = format_str
        self._registry_key = None  # Will be set by registry

    @abstractmethod
    def compute(self, dynamics: Any, step: int, dt: float) -> float:
        """
        Compute the metric value for the current simulation state.

        Args:
            dynamics: Dynamics simulation object containing state and backend
            step: Current simulation step number
            dt: Time step size (for time-dependent metrics)

        Returns:
            Computed metric value as a float
        """
        pass

    def __str__(self):
        """Return string representation of the metric."""
        return f"{self.name}{self.format_str}"

    def to_config(self) -> dict:
        """
        Convert metric to configuration dictionary for serialization.

        Returns:
            Dictionary containing metric configuration
        """
        config = {
            "type": self._registry_key or self.__class__.__name__,
            "name": self.name,
            "format_str": self.format_str,
        }
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Create metric instance from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Metric instance
        """
        # Default implementation - override in subclasses if needed
        return cls()

    @classmethod
    def get_metric_type(cls) -> str:
        """
        Get the metric type identifier for registry.

        Returns:
            Class name as string identifier
        """
        return cls.__name__


class MetricRegistry(BaseRegistry):
    """
    Registry for metric classes to enable dynamic instantiation from YAML.

    This registry extends the BaseRegistry to provide specialized functionality
    for metric creation and configuration parsing. It enables dynamic loading
    of metrics from configuration files and supports both simple and parameterized
    metrics.

    ## Key Features

    - **Dynamic Instantiation**: Create metrics by name from configuration
    - **Parameter Support**: Handle metrics with initialization parameters
    - **YAML Integration**: Parse metric configurations from YAML files
    - **Factory Pattern**: Support custom factory methods for complex metrics
    - **Serialization**: Automatic registry key management for config export

    ## Usage Examples

    ### Basic Metric Creation
    ```python
    # Create simple metric
    energy_metric = MetricRegistry.create_metric("energy")

    # Create parameterized metric
    bond_metric = MetricRegistry.create_metric("bond_length", indices=[0, 1])
    ```

    ### Configuration Parsing
    ```python
    config = [
        "energy",
        "grad_norm",
        {"bond_length": {"indices": [0, 1]}}
    ]
    metrics = MetricRegistry.parse_metrics_config(config)
    ```

    ## Supported Configuration Formats

    The registry supports multiple configuration formats:

    ```yaml
    # Simple metrics
    metrics:
      - energy
      - grad_norm

    # Parameterized metrics
    metrics:
      - bond_length:
          indices: [0, 1]
      - custom_metric:
          param1: value1
          param2: value2
    ```

    ## Integration with OmegaConf

    The registry automatically handles OmegaConf types for YAML parsing,
    making it compatible with Hydra configuration systems.
    """

    # TODO: add a metric parser so that it can create either from string or DictConfig

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Metric:
        """
        Create an instance of a metric by name.

        Args:
            name: Registered name of the metric class
            *args: Positional arguments for metric initialization
            **kwargs: Keyword arguments for metric initialization

        Returns:
            Metric instance
        """
        return cast(Metric, super().create(name, *args, **kwargs))

    @classmethod
    def create_metric(cls, name_or_config: Union[str, dict], **kwargs):
        """
        Create a metric instance by name or config dict with enhanced factory support.

        This method provides additional factory logic for metrics that have
        custom creation methods or require special initialization. It can handle
        both string names and configuration dictionaries.

        Args:
            name_or_config: Either a string metric name or a config dict like {"bond_length": [0, 1]}
            **kwargs: Additional keyword arguments for metric initialization

        Returns:
            Metric instance with registry key set
        """
        if isinstance(name_or_config, str):
            name = name_or_config
            params = kwargs
        elif isinstance(name_or_config, dict):
            if len(name_or_config) != 1:
                raise ValueError("Config dict must have exactly one key-value pair")
            name, config_params = next(iter(name_or_config.items()))
            if isinstance(config_params, dict):
                params = {**config_params, **kwargs}
            elif isinstance(config_params, list):
                params = {"indices": config_params, **kwargs}
            else:
                raise ValueError("Config params must be dict or list")
        else:
            # Handle OmegaConf types
            try:
                import omegaconf

                if isinstance(name_or_config, omegaconf.DictConfig):
                    if len(name_or_config) != 1:
                        raise ValueError(
                            "Config dict must have exactly one key-value pair"
                        )
                    name, config_params = next(iter(name_or_config.items()))
                    if isinstance(config_params, (dict, omegaconf.DictConfig)):
                        params = {**dict(config_params), **kwargs}
                    elif isinstance(config_params, (list, omegaconf.ListConfig)):
                        params = {"indices": list(config_params), **kwargs}
                    else:
                        raise ValueError("Config params must be dict or list")
                else:
                    raise ValueError("name_or_config must be str or dict")
            except ImportError:
                raise ValueError("name_or_config must be str or dict")

        metric_class = cls.get(name)

        # Check if the metric class has a custom factory method
        if hasattr(metric_class, "create_from_kwargs"):
            metric = metric_class.create_from_kwargs(**params)
        else:
            metric = cls.create(name, **params)

        # Set the registry key for serialization
        metric._registry_key = name
        return metric

    @classmethod
    def list_available_metrics(cls):
        """
        List all available metric types registered in the system.

        Returns:
            List of strings containing the names of all registered metric types
            that can be instantiated via the registry.

        Examples:
            ```python
            available = MetricRegistry.list_available_metrics()
            print(available)  # ['energy', 'grad_norm', 'bond_length', 'hess_eval_max', ...]
            ```
        """
        return cls.list_available()

    @classmethod
    def parse_metrics_config(cls, metrics_config) -> list[Metric] | None:
        """
        Parse metrics configuration from YAML and create metric instances using registry.

        This method handles various configuration formats for metrics, supporting both
        simple metric names and parameterized metrics. It integrates with OmegaConf
        for YAML parsing and provides robust error handling for invalid configurations.

        Args:
            metrics_config: Configuration specification for metrics. Can be:
                - None or empty: Returns None
                - List of strings: Simple metric names like ["energy", "grad_norm"]
                - List of dicts: Parameterized metrics like [{"bond_length": {"indices": [0, 1]}}]
                - Mixed list: Combination of strings and dicts

        Returns:
            List of Metric instances, or None if no valid metrics found

        Raises:
            No exceptions raised - invalid configurations are logged and skipped

        Examples:
            ```python
            # Simple metrics
            config = ["energy", "grad_norm"]
            metrics = MetricRegistry.parse_metrics_config(config)

            # Parameterized metrics
            config = [{"bond_length": {"indices": [0, 1]}}]
            metrics = MetricRegistry.parse_metrics_config(config)

            # Mixed configuration
            config = ["energy", {"bond_length": {"indices": [0, 1]}}]
            metrics = MetricRegistry.parse_metrics_config(config)
            ```

        Notes:
            - Invalid metric specifications are logged as warnings and skipped
            - Supports OmegaConf types for Hydra integration
            - Returns None for empty or invalid configurations
            - Metric registry keys are automatically set for serialization
        """
        if not metrics_config:
            return None

        metrics = []
        for metric_spec in metrics_config:
            try:
                if isinstance(metric_spec, str):
                    # Simple metric name
                    metric = cls.create_metric(metric_spec)
                    metrics.append(metric)
                elif isinstance(metric_spec, dict):
                    # Metric with parameters
                    for metric_name, params in metric_spec.items():
                        if isinstance(params, list):
                            # Bond length with indices
                            metric = cls.create_metric(metric_name, indices=params)
                        else:
                            # Other parameterized metrics
                            metric = cls.create_metric(metric_name, **params)
                        metrics.append(metric)
                else:
                    # Handle OmegaConf types
                    try:
                        # Try to convert to regular Python types
                        import omegaconf

                        if isinstance(metric_spec, omegaconf.DictConfig):
                            for metric_name, params in metric_spec.items():
                                if isinstance(params, (list, omegaconf.ListConfig)):
                                    # Bond length with indices
                                    metric = cls.create_metric(
                                        metric_name, indices=list(params)
                                    )
                                else:
                                    # Other parameterized metrics
                                    metric = cls.create_metric(
                                        metric_name, **dict(params)
                                    )
                                metrics.append(metric)
                    except ImportError:
                        pass
            except ValueError as e:
                # Log warning if logger is available, otherwise silently continue
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Warning: {e}")
                except ImportError:
                    pass

        return metrics if metrics else None


@MetricRegistry.register("energy")
class EnergyMetric(Metric):
    """
    Metric for computing the potential energy of the molecular system.

    This metric computes the potential energy of the current molecular state
    using the backend's energy computation method. It's one of the most
    fundamental metrics for monitoring simulation progress and convergence.

    ## Key Features

    - **Potential Energy**: Computes E = E(x) for current coordinates x
    - **Convergence Monitoring**: Energy changes indicate progress toward minima
    - **Units**: Energy values in backend-specific units (typically Hartree or kcal/mol)
    - **Backend Integration**: Uses backend.compute_energy() method

    ## Usage

    The energy metric is automatically included in most dynamics simulations
    and provides essential feedback about the quality of the current molecular
    configuration and the progress of optimization or dynamics.
    """

    def __init__(self):
        """Initialize energy metric with default formatting."""
        super().__init__("Energy")

    def compute(self, dynamics, step: int, dt: float) -> float:
        """
        Compute the potential energy of the current molecular state.

        Args:
            dynamics: Dynamics simulation object with backend and state
            step: Current simulation step (unused for energy computation)
            dt: Time step (unused for energy computation)

        Returns:
            Potential energy value in backend units
        """
        return dynamics.backend.compute_energy(dynamics.state)


@MetricRegistry.register("grad_norm")
class GradNormMetric(Metric):
    """
    Metric for computing the Euclidean norm of the gradient vector.

    This metric computes ||∇E|| (the L2 norm of the gradient) which is a key
    indicator of convergence in optimization and dynamics simulations. When
    the gradient norm becomes small, the system is close to a critical point.

    ## Key Features

    - **Convergence Indicator**: ||∇E|| ≈ 0 indicates critical points
    - **Magnitude Information**: Absolute scale for gradient magnitude
    - **Units**: Gradient norm in energy/coordinates units
    - **Numerical Stability**: Handles zero gradients gracefully

    ## Usage

    ```python
    grad_metric = GradNormMetric()
    grad_norm = grad_metric.compute(dynamics, step, dt)
    if grad_norm < 1e-6:
        print("Converged to critical point!")
    ```

    ## Notes

    - Gradient norm is computed as sqrt(sum(∇E_i²))
    - Small values indicate proximity to critical points
    - Used in convergence criteria for geometry optimization
    """

    def __init__(self):
        """Initialize gradient norm metric with scientific notation formatting."""
        super().__init__("Grad_norm", ":10.3e")

    def compute(self, dynamics, step: int, dt: float) -> float:
        """
        Compute the Euclidean norm of the gradient vector.

        Args:
            dynamics: Dynamics simulation object with backend and state
            step: Current simulation step (unused for gradient computation)
            dt: Time step (unused for gradient computation)

        Returns:
            Euclidean norm of the gradient vector
        """
        grad = dynamics.backend.compute_gradient(dynamics.state)
        return np.linalg.norm(grad)


@MetricRegistry.register("bond_length")
class BondLengthMetric(Metric):
    """
    Metric for monitoring the distance between two atoms (bond length).

    This metric computes the Euclidean distance between two specified atoms
    in the molecular system. It's particularly useful for monitoring bond
    lengths during reactions, geometry optimizations, or dynamics simulations
    where specific atomic distances are of interest.

    ## Key Features

    - **Atom Pair Specification**: Configurable atom indices for distance measurement
    - **Real-time Monitoring**: Tracks bond length changes during simulation
    - **Reaction Analysis**: Useful for studying bond breaking/forming processes
    - **Configuration Support**: Full serialization/deserialization support

    ## Usage Examples

    ### Basic Usage
    ```python
    # Monitor bond between atoms 0 and 1
    bond_metric = BondLengthMetric(indices=(0, 1))
    ```

    ### Configuration
    ```python
    # In YAML configuration
    metrics:
      - bond_length:
          indices: [0, 1]

    # Programmatic creation
    metric = BondLengthMetric.from_config({"indices": [0, 1]})
    ```

    ## Attributes

    - `indices`: Tuple of (atom1_index, atom2_index) specifying the bond
    - `name`: Auto-generated name like "Bond_0-1"

    ## Notes

    - Atom indices are 0-based and correspond to the molecular coordinate array
    - Coordinates are assumed to be in shape (n_atoms, 3)
    - Distance is computed in the same units as the coordinate system
    """

    def __init__(self, indices: Tuple[int, int]):
        """
        Initialize bond length metric for specific atom pair.

        Args:
            indices: Tuple of (atom1_index, atom2_index) for distance measurement
        """
        super().__init__(f"Bond_{indices[0]}-{indices[1]}", ":8.4f")
        self.indices = indices

    def compute(self, dynamics, step: int, dt: float) -> float:
        """
        Compute the bond length between the specified atoms.

        Args:
            dynamics: Dynamics simulation object containing molecular state
            step: Current simulation step (unused for this metric)
            dt: Time step (unused for this metric)

        Returns:
            Bond length in coordinate system units
        """
        coords = dynamics.state.reshape(-1, 3)
        bond_length = np.linalg.norm(coords[self.indices[0]] - coords[self.indices[1]])
        return bond_length

    def to_config(self) -> dict:
        """
        Convert metric to configuration dictionary with bond indices.

        Returns:
            Configuration dictionary including atom indices
        """
        config = super().to_config()
        config["indices"] = list(self.indices)
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Create metric instance from configuration dictionary with bond indices.

        Args:
            config: Configuration dictionary containing 'indices' key

        Returns:
            BondLengthMetric instance
        """
        return cls(tuple(config["indices"]))

    @classmethod
    def create_from_kwargs(cls, **kwargs):
        """
        Create metric from keyword arguments.

        Args:
            **kwargs: Must contain 'indices' key with atom pair

        Returns:
            BondLengthMetric instance

        Raises:
            ValueError: If 'indices' parameter is missing
        """
        if "indices" in kwargs:
            return cls(tuple(kwargs["indices"]))
        else:
            raise ValueError("BondLengthMetric requires 'indices' parameter")


@MetricRegistry.register("hess_eval_max")
class HessEvalMaxMetric(Metric):
    """
    Metric for monitoring the maximum Hessian eigenvalue in the relevant subspace.

    This metric tracks the largest eigenvalue of the Hessian matrix within the
    subspace of interest (typically the first `dynamics.index` eigenvalues).
    For saddle dynamics, this helps monitor the behavior along the unstable
    reaction coordinate.

    ## Key Features

    - **Subspace Monitoring**: Focuses on relevant Hessian eigenvalues only
    - **Index-based Selection**: Uses dynamics.index to determine subspace size
    - **Saddle Detection**: Large positive values indicate unstable directions
    - **Dynamics Integration**: Works with HiSD and other Hessian-based methods

    ## Usage

    Particularly useful in:
    - Saddle point finding algorithms (HiSD)
    - Transition state optimization
    - Analysis of reaction pathways
    - Monitoring convergence in Hessian-based dynamics

    ## Notes

    - Returns 0.0 if no Hessian eigenvalues are available
    - Only considers the first `dynamics.index` eigenvalues
    - Maximum eigenvalue indicates the most unstable direction
    """

    def __init__(self):
        """Initialize maximum Hessian eigenvalue metric."""
        super().__init__("Hess_eval_max", ":10.3e")

    def compute(self, dynamics, step: int, dt: float) -> float:
        """
        Compute the maximum Hessian eigenvalue in the relevant subspace.

        Args:
            dynamics: Dynamics simulation object with hess_evals and index
            step: Current simulation step (unused)
            dt: Time step (unused)

        Returns:
            Maximum eigenvalue in the subspace, or 0.0 if unavailable
        """
        if len(dynamics.hess_evals) == 0:
            return 0.0
        # Only consider first dynamics.index eigenvalues
        eigenvals = dynamics.hess_evals[: dynamics.index]
        return np.max(eigenvals) if len(eigenvals) > 0 else 0.0


@MetricRegistry.register("hess_eval_min")
class HessEvalMinMetric(Metric):
    """
    Metric for monitoring the minimum Hessian eigenvalue in the relevant subspace.

    This metric tracks the smallest eigenvalue of the Hessian matrix within the
    subspace of interest. For minimum finding, this should approach zero from
    above, while for saddle finding it helps monitor the stability of the
    optimization.

    ## Key Features

    - **Subspace Monitoring**: Focuses on relevant Hessian eigenvalues only
    - **Index-based Selection**: Uses dynamics.index to determine subspace size
    - **Stability Indicator**: Small positive values indicate near-critical points
    - **Dynamics Integration**: Works with Hessian-based optimization methods

    ## Usage

    Particularly useful in:
    - Geometry optimization convergence monitoring
    - Saddle point characterization
    - Analysis of vibrational frequencies
    - Hessian-based dynamics algorithms

    ## Notes

    - Returns 0.0 if no Hessian eigenvalues are available
    - Only considers the first `dynamics.index` eigenvalues
    - Minimum eigenvalue indicates the most stable direction
    - For minima: eigenvalue → 0⁺, for saddles: eigenvalue → 0⁻
    """

    def __init__(self):
        """Initialize minimum Hessian eigenvalue metric."""
        super().__init__("Hess_eval_min", ":10.3e")

    def compute(self, dynamics, step: int, dt: float) -> float:
        """
        Compute the minimum Hessian eigenvalue in the relevant subspace.

        Args:
            dynamics: Dynamics simulation object with hess_evals and index
            step: Current simulation step (unused)
            dt: Time step (unused)

        Returns:
            Minimum eigenvalue in the subspace, or 0.0 if unavailable
        """
        if len(dynamics.hess_evals) == 0:
            return 0.0
        # Only consider first dynamics.index eigenvalues
        eigenvals = dynamics.hess_evals[: dynamics.index]
        return np.min(eigenvals) if len(eigenvals) > 0 else 0.0
