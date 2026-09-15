"""
Registry pattern implementation for DFTScape component management.

This module provides a flexible registry system for managing and instantiating
DFTScape components dynamically. The BaseRegistry class implements a generic
factory pattern that allows components to be registered by name and later
retrieved or instantiated on demand.

The registry system is designed to support:
- **Plugin Architecture**: Easy registration of new component implementations
- **Factory Pattern**: Centralized component instantiation with dependency injection
- **Dynamic Loading**: Runtime component discovery and loading
- **Type Safety**: Generic type support for compile-time type checking
- **Error Handling**: Clear error messages for missing or invalid components

Key Features:
- Decorator-based registration for clean component declaration
- Automatic subclass isolation (each registry subclass has its own namespace)
- Type-safe generic factory methods
- Comprehensive error reporting with available options
- Thread-safe registration (but not instantiation)

Typical Usage:
    >>> from dftscape.core.registry import BaseRegistry
    >>>
    >>> # Create a component registry
    >>> class BackendRegistry(BaseRegistry):
    ...     pass
    >>>
    >>> # Register components
    >>> @BackendRegistry.register("psi4")
    ... class Psi4Backend:
    ...     def __init__(self, method="hf", basis="sto-3g"):
    ...         self.method = method
    ...         self.basis = basis
    >>>
    >>> @BackendRegistry.register("pyscf")
    ... class PySCFBackend:
    ...     def __init__(self, method="hf"):
    ...         self.method = method
    >>>
    >>> # Use the registry
    >>> backend_class = BackendRegistry.get("psi4")
    >>> backend_instance = BackendRegistry.create("psi4", method="b3lyp", basis="6-31g")
    >>> available = BackendRegistry.list_available()
    >>> print(f"Available backends: {available}")

Integration Points:
- Used by Backend system for quantum chemistry engine management
- Integrated with Distance function registries for metric selection
- Consumed by Orchestrator for dynamic component loading
- Supports plugin systems for extending DFTScape functionality

Implementation Notes:
- Each BaseRegistry subclass gets its own isolated registry dictionary
- Registration is thread-safe but should be done at import time
- Component instantiation is not thread-safe (depends on component implementation)
- Memory usage scales with number of registered components
- Registry lookups are O(1) for fast component access

Notes:
- Component names should be unique within each registry
- Registration typically happens at module import time
- Error messages include available options for better debugging
- Generic type support enables IDE autocompletion and type checking
"""

from __future__ import annotations
from typing import TypeVar, Type, Callable

T = TypeVar("T")


class BaseRegistry:
    """
    Generic registry pattern implementation for DFTScape component management.

    This base class provides a flexible factory pattern for registering, retrieving,
    and instantiating components dynamically. Each subclass automatically gets its
    own isolated registry namespace, allowing multiple component types to coexist
    without naming conflicts.

    The registry uses a decorator-based registration system and provides type-safe
    factory methods for component instantiation. It's designed to support plugin
    architectures and dynamic component loading in DFTScape.

    Attributes:
        _registry (dict[str, Type[T]]): Class-level dictionary mapping component
            names to their corresponding classes. Each BaseRegistry subclass has
            its own isolated registry dictionary.

    Class Methods:
        register: Decorator for registering component classes
        get: Retrieve component class by name
        create: Instantiate component by name with arguments
        list_available: Get list of all registered component names

    Design Patterns:
        - **Registry Pattern**: Centralized component management and discovery
        - **Factory Pattern**: Type-safe component instantiation
        - **Decorator Pattern**: Clean component registration syntax
        - **Subclass Isolation**: Each registry subclass has independent namespace

    Thread Safety:
        - Registration is thread-safe (done at import time)
        - Instantiation is not thread-safe (depends on component implementation)
        - Registry lookups are thread-safe (dict access)

    Performance Characteristics:
        - Registration: O(1) time complexity
        - Lookup: O(1) time complexity
        - Memory: O(n) where n is number of registered components

    Usage Examples:
        >>> # Create a custom registry
        >>> class CalculatorRegistry(BaseRegistry):
        ...     pass
        >>>
        >>> # Register components
        >>> @CalculatorRegistry.register("basic")
        ... class BasicCalculator:
        ...     def add(self, a, b): return a + b
        >>>
        >>> @CalculatorRegistry.register("scientific")
        ... class ScientificCalculator(BasicCalculator):
        ...     def sin(self, x): return math.sin(x)
        >>>
        >>> # Use the registry
        >>> calc_class = CalculatorRegistry.get("scientific")
        >>> calc = CalculatorRegistry.create("basic")
        >>> print(CalculatorRegistry.list_available())

    Error Handling:
        - Raises ValueError for unknown component names
        - Error messages include available options for debugging
        - Type-safe generic methods prevent runtime type errors

    Integration Notes:
        - Used by DFTScape's backend system for quantum chemistry engines
        - Supports distance function registries for metric selection
        - Enables plugin architecture for extending DFTScape functionality
        - Consumed by orchestrator for dynamic component loading

    Implementation Details:
        - Uses __init_subclass__ for automatic registry initialization
        - Generic type support with TypeVar for compile-time type checking
        - Decorator returns the original class for transparent registration
        - Registry dictionary is per-subclass to avoid naming conflicts
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own registry dictionary
        cls._registry = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a component class with a unique name.

        This method returns a decorator that can be applied to component classes
        to register them in the registry. The registration happens at import time
        and makes the component discoverable by name.

        Args:
            name (str): Unique identifier for the component. Must be unique within
                this registry subclass. Should use lowercase with underscores for
                consistency (e.g., "psi4_backend", "distance_euclidean").

        Returns:
            Callable[[Type[T]], Type[T]]: Decorator function that registers the
                component class and returns it unchanged.

        Raises:
            No exceptions are raised during registration. Duplicate names will
            silently overwrite previous registrations.

        Examples:
            >>> class BackendRegistry(BaseRegistry):
            ...     pass
            >>>
            >>> @BackendRegistry.register("psi4")
            ... class Psi4Backend:
            ...     def compute_energy(self, molecule):
            ...         return 0.0
            >>>
            >>> # Component is now registered and discoverable
            >>> backend_class = BackendRegistry.get("psi4")

        Notes:
            - Registration is typically done at module import time
            - The decorator returns the original class unchanged
            - Duplicate names overwrite previous registrations
            - Thread-safe when done at import time
            - Component names should follow consistent naming conventions
        """

        def decorator(component_class: Type[T]) -> Type[T]:
            cls._registry[name] = component_class
            return component_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[T]:
        """
        Retrieve a registered component class by its name.

        This method performs a lookup in the registry and returns the component
        class if found. It's the primary method for accessing registered components
        without instantiation.

        Args:
            name (str): The name of the component to retrieve. Case-sensitive and
                must match the registration name exactly.

        Returns:
            Type[T]: The component class registered under the given name.

        Raises:
            ValueError: If the component name is not found in the registry.
                The error message includes all available component names for
                easier debugging.

        Examples:
            >>> class MetricRegistry(BaseRegistry):
            ...     pass
            >>>
            >>> @MetricRegistry.register("euclidean")
            ... class EuclideanDistance:
            ...     @staticmethod
            ...     def compute(a, b):
            ...         return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            >>>
            >>> # Get the class (not an instance)
            >>> distance_class = MetricRegistry.get("euclidean")
            >>> distance_func = distance_class.compute
            >>>
            >>> # Error handling
            >>> try:
            ...     unknown = MetricRegistry.get("manhattan")
            ... except ValueError as e:
            ...     print(f"Error: {e}")

        Performance:
            - O(1) time complexity due to dictionary lookup
            - Thread-safe for read operations
            - Memory efficient (no object instantiation)

        Notes:
            - Use this method when you need the class itself, not an instance
            - Combine with create() for instantiation with parameters
            - Error messages are designed to help with debugging
            - Case-sensitive name matching
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown {cls.__name__} '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        """
        Create and return a new instance of a registered component.

        This factory method combines component lookup with instantiation, providing
        a convenient way to create component instances with arbitrary arguments.
        It's the primary method for dynamic component instantiation in DFTScape.

        Args:
            name (str): The name of the component to instantiate. Must be registered
                in this registry subclass.
            *args: Positional arguments to pass to the component constructor.
            **kwargs: Keyword arguments to pass to the component constructor.

        Returns:
            T: A new instance of the requested component class, initialized with
                the provided arguments.

        Raises:
            ValueError: If the component name is not found in the registry.
                The error message includes all available component names.
            TypeError: If the component constructor arguments are invalid.
                This depends on the specific component's __init__ signature.
            Any exception raised by the component's constructor.

        Examples:
            >>> class BackendRegistry(BaseRegistry):
            ...     pass
            >>>
            >>> @BackendRegistry.register("psi4")
            ... class Psi4Backend:
            ...     def __init__(self, method="hf", basis="sto-3g", memory="500MB"):
            ...         self.method = method
            ...         self.basis = basis
            ...         self.memory = memory
            >>>
            >>> # Create instances with different configurations
            >>> backend1 = BackendRegistry.create("psi4")
            >>> backend2 = BackendRegistry.create("psi4", method="b3lyp", basis="6-31g")
            >>> backend3 = BackendRegistry.create("psi4", "ccsd", "cc-pvdz", memory="2GB")
            >>>
            >>> # Error handling
            >>> try:
            ...     unknown = BackendRegistry.create("unknown_backend")
            ... except ValueError as e:
            ...     print(f"Available backends: {BackendRegistry.list_available()}")

        Performance:
            - O(1) lookup time plus component instantiation time
            - Memory usage depends on the component instance
            - Not thread-safe (depends on component implementation)

        Notes:
            - This is the recommended way to create component instances
            - Arguments are passed directly to the component constructor
            - Component instantiation errors are not caught (by design)
            - Useful for dependency injection and plugin systems
            - Supports both positional and keyword arguments
        """
        component_class = cls.get(name)
        return component_class(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """
        Return a list of all currently registered component names.

        This method provides introspection capabilities for the registry, allowing
        users and systems to discover what components are available at runtime.
        Useful for debugging, UI generation, and dynamic feature detection.

        Returns:
            list[str]: A list of all component names currently registered in this
                registry subclass. The list is a copy of the registry keys, so
                modifications won't affect the original registry.

        Examples:
            >>> class AlgorithmRegistry(BaseRegistry):
            ...     pass
            >>>
            >>> @AlgorithmRegistry.register("gradient_descent")
            ... class GradientDescent:
            ...     pass
            >>>
            >>> @AlgorithmRegistry.register("newton_method")
            ... class NewtonMethod:
            ...     pass
            >>>
            >>> # List all available algorithms
            >>> algorithms = AlgorithmRegistry.list_available()
            >>> print(f"Available algorithms: {algorithms}")
            >>> print(f"Number of algorithms: {len(algorithms)}")
            >>>
            >>> # Use for validation
            >>> def validate_algorithm(name):
            ...     if name not in AlgorithmRegistry.list_available():
            ...         raise ValueError(f"Unknown algorithm: {name}")
            ...     return True

        Performance:
            - O(n) time complexity where n is number of registered components
            - O(n) memory usage for the returned list
            - Thread-safe for read operations

        Notes:
            - Returns a new list each time (safe to modify)
            - Useful for generating UI options or configuration choices
            - Can be used for validation and error handling
            - Order is not guaranteed (dict iteration order)
            - Empty list returned if no components are registered
        """
        return list(cls._registry.keys())
