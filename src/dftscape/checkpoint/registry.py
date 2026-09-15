from __future__ import annotations
from typing import cast, Dict, Any, Protocol

from ..common import BaseRegistry


# ------------------------------------------------------------------ #
#                           Registry Pattern                         #
# ------------------------------------------------------------------ #


class Checkpointer(Protocol):

    def save(self, path: str, payload: Dict[str, Any]) -> None: ...

    def load(self, path: str) -> Dict[str, Any]: ...


class CheckpointRegistry(BaseRegistry):

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Checkpointer:
        return cast(Checkpointer, super().create(name, *args, **kwargs))


# Import implementations to trigger decorator registration
try:
    from . import impl_npz  # noqa: F401
except ImportError:
    pass

try:
    from . import impl_json  # noqa: F401
except ImportError:
    pass
