from __future__ import annotations
from typing import Protocol, Any, Dict, Optional, TYPE_CHECKING, runtime_checkable
import numpy as np

if TYPE_CHECKING:
    from .search_primitives import RunCondition
    from .point import CriticalPoint


@runtime_checkable
class Backend(Protocol):

    def compute_energy(self, state: np.ndarray) -> float:
        ...

    def compute_gradient(self, state: np.ndarray) -> np.ndarray:
        ...

    def compute_hessian(self, state: np.ndarray) -> np.ndarray:
        ...

    def get_xyz(self, state: np.ndarray) -> str:
        ...


@runtime_checkable
class HiSDRunner(Protocol):

    def run(self, run_condition: "RunCondition") -> Optional["CriticalPoint"]:
        ...


@runtime_checkable
class Checkpointer(Protocol):

    def save(self, path: str, payload: Dict[str, Any]) -> None:
        ...

    def load(self, path: str) -> Dict[str, Any]:
        ...
