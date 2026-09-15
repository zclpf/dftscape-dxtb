"""
DFTScape Checkpoint Subpackage
=============================

The checkpoint subpackage provides a unified, extensible framework for data
persistence and recovery in DFTScape. It supports multiple storage formats
through a pluggable architecture, enabling efficient checkpointing of
molecular simulation data including coordinates, energies, forces, and
other numerical arrays.

Key Features
============

Multiple Storage Formats:
    - NPZ: High-performance compressed NumPy arrays with type preservation
    - JSON: Human-readable text format for configuration and metadata
    - Extensible: Easy to add new checkpoint formats via registry pattern

Type Preservation:
    - Exact preservation of NumPy dtypes and array shapes
    - Scalar type restoration (int, float, bool, str)
    - Cross-platform compatibility with byte order normalization

Performance Optimized:
    - Compressed storage to reduce disk usage
    - Efficient I/O for large numerical datasets
    - Memory-efficient loading with lazy evaluation support

Registry-Based Architecture:
    - Pluggable checkpoint implementations
    - Runtime format selection
    - Consistent API across all formats

Available Implementations
========================

NPZCheckpointer:
    Best for large numerical datasets, molecular coordinates, forces, and
    Hessians. Provides exact type preservation and compression.

JSONCheckpointer:
    Best for configuration data, metadata, and human-readable storage.
    Supports complex nested structures with full type fidelity.

Usage Examples
==============

Basic Checkpointing:
    >>> from dftscape.checkpoint import CheckpointRegistry
    >>> # Create NPZ checkpointer for numerical data
    >>> ckpt = CheckpointRegistry.create("npz")
    >>> data = {"coords": np.array([[1, 2, 3]]), "energy": -123.45}
    >>> ckpt.save("checkpoint.npz", data)
    >>> restored = ckpt.load("checkpoint.npz")

Format Selection:
    >>> # Use JSON for configuration data
    >>> json_ckpt = CheckpointRegistry.create("json")
    >>> config = {"method": "DFT", "basis": "6-31G*", "converged": True}
    >>> json_ckpt.save("config.json", config)

Explicit Array/Scalar Separation:
    >>> # For fine-grained control over data types
    >>> arrays = {"forces": np.array([[0.1, 0.2, 0.3]])}
    >>> scalars = {"step": 100, "energy": -123.45}
    >>> ckpt.save_arrays_and_scalars("data.npz", arrays, scalars)
    >>> arrays_loaded, scalars_loaded = ckpt.load_arrays_and_scalars("data.npz")

Registry Operations:
    >>> # List available checkpoint formats
    >>> available = CheckpointRegistry.list_available()
    >>> print(available)  # ['npz', 'json']
    >>>
    >>> # Get checkpointer instance by name
    >>> npz_ckpt = CheckpointRegistry.get("npz")

API Overview
============

CheckpointRegistry:
    Central registry for checkpoint implementations
    - create(format): Create checkpointer instance
    - get(format): Get checkpointer class
    - list_available(): List registered formats
    - register(name): Decorator for registering new formats

Checkpointer Protocol:
    Common interface for all checkpoint implementations
    - save(path, data): Save unified dictionary
    - load(path): Load unified dictionary
    - save_arrays_and_scalars(path, arrays, scalars): Save with separation
    - load_arrays_and_scalars(path): Load with separation

Notes
=====

- All checkpointers support automatic directory creation
- NPZ format provides best performance for numerical data
- JSON format provides best readability for configuration
- Registry pattern enables easy extension with new formats
- All implementations are thread-safe for read operations
- Type preservation ensures exact numerical fidelity

See Also
========

- dftscape.checkpoint.registry: Registry implementation and protocol
- dftscape.checkpoint.impl_npz: NPZ-based high-performance checkpointer
- dftscape.checkpoint.impl_json: JSON-based human-readable checkpointer
"""

from .registry import CheckpointRegistry

__all__ = ["CheckpointRegistry"]
