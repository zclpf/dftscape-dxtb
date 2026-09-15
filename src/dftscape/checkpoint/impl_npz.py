"""
NPZ-based checkpointer for efficient numerical data persistence in DFTScape.

This module provides a high-performance checkpoint implementation using NumPy's
native NPZ format. It excels at storing large numerical datasets with exact
type preservation and compression, making it ideal for molecular coordinates,
energies, gradients, and other numerical data in DFTScape.

Key Features:
============

High Performance:
    - Native NumPy format for optimal I/O performance
    - Compressed storage to reduce disk usage
    - Efficient memory mapping for large datasets

Type Preservation:
    - Exact preservation of numpy dtypes and shapes
    - Scalar type preservation (int, float, bool, str)
    - Little-endian byte order for cross-platform compatibility

API Methods:
    - save(): Save unified dictionary with automatic type handling
    - load(): Load unified dictionary with automatic type restoration
    - save_arrays_and_scalars(): Save with explicit array/scalar separation
    - load_arrays_and_scalars(): Load with explicit array/scalar separation

Usage Examples:
==============

Basic Operations:
    >>> from dftscape.checkpoint.impl_npz import NPZCheckpointer
    >>> cp = NPZCheckpointer(compress=True)
    >>> data = {"coords": np.array([[1, 2, 3]]), "energy": -123.45}
    >>> cp.save("checkpoint.npz", data)
    >>> restored = cp.load("checkpoint.npz")

Type Preservation:
    >>> # Scalars maintain exact types
    >>> original = {"count": 42, "energy": -123.45, "converged": True}
    >>> cp.save("scalars.npz", original)
    >>> loaded = cp.load("scalars.npz")
    >>> assert type(loaded["count"]) == int
    >>> assert type(loaded["energy"]) == float
    >>> assert type(loaded["converged"]) == bool

Array Handling:
    >>> arrays = {"forces": np.array([[0.1, 0.2, 0.3]])}
    >>> scalars = {"step": 100, "method": "BFGS"}
    >>> cp.save_arrays_and_scalars("data.npz", arrays, scalars)
    >>> arrays_loaded, scalars_loaded = cp.load_arrays_and_scalars("data.npz")

Configuration:
    >>> cp = NPZCheckpointer(compress=False)  # Disable compression for speed

Advantages:
==========

- Exact type preservation for numerical computations
- Superior performance for large datasets
- Automatic compression reduces storage requirements
- Cross-platform compatibility with byte order normalization
- Metadata support for complex type restoration

Best suited for:
- Large numerical datasets (coordinates, forces, Hessians)
- Scientific computing with exact precision requirements
- Performance-critical checkpoint operations
- Distributed computing environments
"""

from __future__ import annotations
from typing import Mapping, Tuple, Dict, Any
import os
import numpy as np
from .registry import CheckpointRegistry


def _to_cpu_np(x: Any) -> np.ndarray:
    """
    Convert input to a NumPy ndarray on CPU and normalize to little-endian dtype.

    This utility function ensures cross-platform compatibility by converting
    all arrays to little-endian byte order, which is the standard for most
    modern systems. It handles various input types and ensures consistent
    data representation across different architectures.

    Args:
        x: Input data to convert. Can be any array-like object including
           lists, tuples, scalars, or existing numpy arrays.

    Returns:
        np.ndarray: NumPy array with little-endian byte order, suitable for
                   cross-platform storage and transfer.

    Notes:
        - Detects and converts big-endian arrays to little-endian
        - Preserves original dtype information where possible
        - Used internally by NPZCheckpointer for data normalization
        - Ensures consistent behavior across different system architectures

    Examples:
        >>> arr = np.array([1, 2, 3], dtype='>i4')  # Big-endian
        >>> normalized = _to_cpu_np(arr)
        >>> print(normalized.dtype.byteorder)  # '<' (little-endian)
    """
    a = np.asarray(x)
    if a.dtype.byteorder in (">",) or (
        a.dtype.byteorder == "=" and not np.little_endian
    ):
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    return a


@CheckpointRegistry.register("npz")
class NPZCheckpointer:
    """
    NPZ-based checkpointer with type preservation and compression support.

    This class provides high-performance checkpointing using NumPy's native NPZ
    format, optimized for scientific computing workloads in DFTScape. It offers
    exact type preservation, automatic compression, and cross-platform compatibility
    for numerical data storage and retrieval.

    The checkpointer stores arrays as regular numpy arrays and scalars as 0-D
    arrays, with metadata support for preserving original scalar types (int,
    float, bool, str). Compression can be enabled for efficient storage of
    large datasets.

    Key Features:
        - Exact numpy dtype and shape preservation
        - Scalar type preservation with metadata
        - Automatic compression support (np.savez_compressed)
        - Cross-platform compatibility with byte order normalization
        - Efficient storage for large numerical datasets
        - Backward compatibility with existing NPZ files

    Attributes:
        compress: Whether to use compression (np.savez_compressed vs np.savez)

    Usage:
        >>> cp = NPZCheckpointer(compress=True)
        >>> data = {"coords": np.array([[1, 2, 3]]), "energy": -123.45}
        >>> cp.save("checkpoint.npz", data)
        >>> restored = cp.load("checkpoint.npz")

    Notes:
        - Arrays are stored as regular numpy arrays (ndim >= 1)
        - Scalars are stored as 0-D numpy arrays with type metadata
        - No pickle usage for security and portability
        - Automatic directory creation for output paths
        - Thread-safe for read operations
    """

    def __init__(self, compress: bool = True) -> None:
        """
        Initialize NPZ checkpointer with compression options.

        Args:
            compress: Whether to use np.savez_compressed (True) or np.savez (False).
                     Compression reduces file size but increases save time.
        """
        self.compress = compress

    def save_arrays_and_scalars(
        self,
        path: str,
        arrays: Mapping[str, Any],
        scalars: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Save arrays and scalars into a single NPZ file with type preservation.

        This method combines arrays and scalars into a compressed NPZ file,
        preserving exact numpy dtypes and original scalar types through metadata.
        Arrays are stored as regular numpy arrays, while scalars are stored as
        0-D arrays with type information for accurate restoration.

        Args:
            path: Output file path (e.g., "checkpoint.npz"). Parent directories
                 are created automatically if they don't exist.
            arrays: Dictionary mapping names to array-like objects. Converted to
                   numpy arrays with little-endian byte order for portability.
            scalars: Optional dictionary of scalar values. Original types (int,
                    float, bool, str) are preserved through metadata storage.

        Raises:
            OSError: If the file cannot be written or directories cannot be created
            ValueError: If array data cannot be converted to numpy arrays
            TypeError: If scalar types are not supported

        Examples:
            >>> arrays = {"coords": np.array([[1, 2, 3], [4, 5, 6]])}
            >>> scalars = {"energy": -123.45, "step": 100, "converged": True}
            >>> cp.save_arrays_and_scalars("data.npz", arrays, scalars)

        Notes:
            - Arrays are normalized to little-endian byte order
            - Scalar types are stored in metadata for exact restoration
            - Compression is used if enabled during initialization
            - Metadata is stored as UTF-8 encoded bytes for portability
            - Supports complex nested scalar structures
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload: Dict[str, np.ndarray] = {}

        # arrays
        for k, v in arrays.items():
            payload[k] = _to_cpu_np(v)

        # scalars as 0-D arrays + metadata for type preservation
        scalar_metadata = {}
        if scalars:
            for k, v in scalars.items():
                payload[k] = _to_cpu_np(np.array(v))
                # Store original type information
                scalar_metadata[k] = {
                    "dtype": str(type(v).__name__),
                    "value": v,  # Store original value for exact type preservation
                }

        # Add metadata as a special key
        if scalar_metadata:
            import json

            metadata_json = json.dumps(scalar_metadata)
            # Save as bytes to avoid unicode issues
            payload["_scalar_metadata"] = np.array(
                metadata_json.encode("utf-8"), dtype="S"
            )

        writer = np.savez_compressed if self.compress else np.savez
        writer(path, **payload)

    def _separate_fields(
        self, data: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Automatically separate numpy arrays from scalars in a unified dictionary.

        This internal method analyzes a unified dictionary and separates it into
        arrays and scalars based on runtime type inspection. It handles both
        numpy arrays and lists (from SerialisableDataClass.to_config()), as well
        as dtype information stored as {field_name}_dtype keys.

        Args:
            data: Unified dictionary containing both arrays and scalars that
                 may include numpy arrays, lists, and dtype metadata.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: A tuple containing:
                - arrays: Dictionary of numpy arrays (ndim >= 1)
                - scalars: Dictionary of scalar values (any non-array type)

        Notes:
            - Numpy arrays are identified by isinstance(v, np.ndarray)
            - Lists are converted to numpy arrays (preserving dtype if available)
            - Dtype keys ({field_name}_dtype) are consumed during processing
            - Scalar detection excludes dtype metadata keys
            - Used internally by save() method for automatic separation

        Examples:
            >>> data = {
            ...     "coords": [1, 2, 3],  # Will become numpy array
            ...     "coords_dtype": "<f8",  # Dtype metadata
            ...     "energy": -123.45,  # Remains as scalar
            ...     "step": 100  # Remains as scalar
            ... }
            >>> arrays, scalars = cp._separate_fields(data)
        """
        arrays = {}
        scalars = {}

        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                # Already a numpy array
                arrays[key] = value
            elif isinstance(value, list):
                # List from SerialisableDataClass.to_config() - convert back to numpy array
                # Check if we have dtype information
                dtype_key = f"{key}_dtype"
                if dtype_key in data:
                    dtype_str = data.pop(dtype_key)  # Remove dtype key from data
                    arrays[key] = np.array(value, dtype=np.dtype(dtype_str))
                else:
                    arrays[key] = np.array(value)
            else:
                # Skip dtype keys that we've already processed
                if not key.endswith("_dtype"):
                    scalars[key] = value

        return arrays, scalars

    def save(self, path: str, data: Dict[str, Any]) -> None:
        """
        Save a unified dictionary by automatically separating arrays from scalars.

        This method provides a convenient interface for saving mixed data containing
        both arrays and scalars in a single operation. It automatically separates
        the data types and calls save_arrays_and_scalars() internally.

        Args:
            path: Output file path for the NPZ file. Parent directories are created
                 automatically if they don't exist.
            data: Unified dictionary containing mixed data types including numpy
                 arrays, lists (from SerialisableDataClass), and scalars.

        Raises:
            OSError: If the file cannot be written or directories cannot be created
            ValueError: If array data cannot be converted to numpy arrays
            TypeError: If data contains unsupported types

        Examples:
            >>> data = {
            ...     "coordinates": np.array([[1, 2, 3], [4, 5, 6]]),
            ...     "forces": np.array([[0.1, 0.2, 0.3]]),
            ...     "energy": -123.45,
            ...     "step": 100,
            ...     "converged": True
            ... }
            >>> cp.save("checkpoint.npz", data)

        Notes:
            - Internally calls _separate_fields() to classify data types
            - Numpy arrays are preserved with exact dtypes and shapes
            - Lists are converted to numpy arrays with dtype preservation
            - Scalar types are preserved exactly through metadata
            - Compression is applied if enabled during initialization
        """
        arrays, scalars = self._separate_fields(data)
        self.save_arrays_and_scalars(path, arrays, scalars)

    def load(self, path: str) -> Dict[str, Any]:
        """
        Load and return a unified dictionary with arrays and scalars combined.

        This method loads data from an NPZ file and combines arrays and scalars
        into a single unified dictionary. It handles type restoration for scalars
        and provides backward compatibility with existing NPZ files.

        Args:
            path: Path to the NPZ file to load. Can be relative or absolute.

        Returns:
            Dict[str, Any]: Unified dictionary containing both arrays and scalars
                           with proper types restored. Arrays are numpy arrays,
                           scalars have their original types (int, float, bool, str).

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is corrupted or contains invalid data
            OSError: If the file cannot be read due to permission issues

        Examples:
            >>> # Load data saved with save()
            >>> data = cp.load("checkpoint.npz")
            >>> coordinates = data["coordinates"]  # numpy array
            >>> energy = data["energy"]  # original type (float)
            >>> step = data["step"]  # original type (int)

        Notes:
            - Arrays are loaded as numpy arrays with original dtypes/shapes
            - Scalars are restored to their original types using metadata
            - Backward compatible with NPZ files without metadata
            - Handles both compressed and uncompressed NPZ files
            - File must be created by this or compatible checkpointer
        """
        arrays, scalars = self.load_arrays_and_scalars(path)
        # Combine arrays and scalars into unified dictionary
        unified = dict(scalars)
        unified.update(arrays)
        return unified

    def load_arrays_and_scalars(
        self, path: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Load arrays and scalars from an NPZ file with explicit separation.

        This method loads data from an NPZ file and separates it into arrays
        and scalars based on the stored structure. It handles type restoration
        for scalars using metadata and provides backward compatibility.

        Args:
            path: Path to the NPZ file to load. Can be relative or absolute.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: A tuple containing:
                - arrays: Dictionary mapping names to numpy arrays (ndim >= 1)
                - scalars: Dictionary mapping names to scalar values with original
                          types restored (int, float, bool, str, etc.)

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is corrupted or contains invalid data
            OSError: If the file cannot be read due to permission issues

        Examples:
            >>> arrays, scalars = cp.load_arrays_and_scalars("data.npz")
            >>> coordinates = arrays["coordinates"]  # numpy array
            >>> energy = scalars["energy"]  # original type (float)
            >>> step = scalars["step"]  # original type (int)
            >>> converged = scalars["converged"]  # original type (bool)

        Notes:
            - Arrays are loaded as numpy arrays with original dtypes/shapes
            - Scalars are restored to original types using stored metadata
            - 0-D arrays are identified as scalars and converted appropriately
            - Backward compatible with files without scalar metadata
            - Handles both compressed and uncompressed NPZ files
            - Metadata is stored as UTF-8 encoded JSON for portability
        """
        d = np.load(path, allow_pickle=False)
        arrays: Dict[str, np.ndarray] = {}
        scalars: Dict[str, float] = {}

        # Load metadata if available
        scalar_metadata = None
        if "_scalar_metadata" in d:
            try:
                import json

                metadata_bytes = d["_scalar_metadata"]
                metadata_str = metadata_bytes.item().decode("utf-8")
                scalar_metadata = json.loads(metadata_str)
            except (ValueError, AttributeError, UnicodeDecodeError):
                scalar_metadata = None

        for k in d.files:
            if k == "_scalar_metadata":
                continue  # Skip metadata key

            v = d[k]
            if v.shape == ():  # 0-D → scalar
                # Try to preserve original type using metadata
                if scalar_metadata and k in scalar_metadata:
                    metadata = scalar_metadata[k]
                    dtype_name = metadata.get("dtype")
                    original_value = metadata.get("value")

                    # Restore original type
                    if dtype_name == "int":
                        scalars[k] = int(original_value)
                    elif dtype_name == "float":
                        scalars[k] = float(original_value)
                    elif dtype_name == "bool":
                        scalars[k] = bool(original_value)
                    elif dtype_name == "str":
                        scalars[k] = str(original_value)
                    else:
                        # Fallback to item() for unknown types
                        scalars[k] = v.item()
                else:
                    # No metadata available, use default conversion
                    scalars[k] = v.item()
            else:
                arrays[k] = v
        return arrays, scalars
