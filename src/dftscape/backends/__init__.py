from .registry import BackendRegistry
from .impl_psi4 import Psi4Backend  # Import to trigger decorator registration
from .impl_dxtb import DxtbBackend 
from .impl_lj import LennardJonesBackend 

# Import test backend (requires JAX)
try:
    from .impl_test import (
        TestBackend,
    )  # Import to trigger decorator registration

    _TEST_AVAILABLE = True
except ImportError:
    _TEST_AVAILABLE = False
    TestBackend = None

from ..core.interfaces import Backend

__all__ = [
    "Backend",
    "BackendRegistry",
    "Psi4Backend",
    "TestBackend",
    "DxtbBackend",
    "LennardJonesBackend"
]
