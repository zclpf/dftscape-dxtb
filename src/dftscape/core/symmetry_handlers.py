from __future__ import annotations
from typing import cast, Protocol

import numpy as np
from ..common import BaseRegistry


class SymmetryHandler(Protocol):

    def get_constraint_subspace(self, state: np.ndarray) -> np.ndarray | None:
        pass

    def validate_subspace(self, subspace: np.ndarray, state: np.ndarray) -> bool:
        pass

    def get_name(self) -> str:
        pass


class SymmetryRegistry(BaseRegistry):

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> SymmetryHandler:
        return cast(SymmetryHandler, super().create(name, *args, **kwargs))


@SymmetryRegistry.register("none")
class NoSymmetryHandler(SymmetryHandler):

    def get_constraint_subspace(self, state: np.ndarray) -> np.ndarray | None:
        return None

    def validate_subspace(self, subspace: np.ndarray, state: np.ndarray) -> bool:
        return True

    def get_name(self) -> str:
        return "No Symmetry"


@SymmetryRegistry.register("molecular")
class MolecularSymmetryHandler(SymmetryHandler):

    def __init__(self, overlap_threshold: float = 0.2):
        self.overlap_threshold = overlap_threshold
        self.overlap_threshold = overlap_threshold

    def get_constraint_subspace(self, state: np.ndarray) -> np.ndarray:
        return self._compute_trivial_subspace(state)

    def validate_subspace(self, subspace: np.ndarray, state: np.ndarray) -> bool:
        trivial_subspace = self.get_constraint_subspace(state)
        if trivial_subspace.size == 0:
            return True

        overlap = np.max(np.linalg.norm(trivial_subspace.T @ subspace, axis=0))
        return overlap < self.overlap_threshold

    def get_name(self) -> str:
        return "Molecular (Translation + Rotation)"

    def _compute_trivial_subspace(self, state: np.ndarray) -> np.ndarray:
        num_atoms = state.size // 3
        atomic_coordinates = state.reshape(-1, 3)
        trivial_ev = np.zeros((6, num_atoms * 3))

        for i in range(3):
            trivial_ev[i, i::3] = 1.0

        # Rotations (Eq. 5): rᵢ = eᵢ × rⱼ for each atom j
        x, y, z = atomic_coordinates[:, 0], atomic_coordinates[:, 1], atomic_coordinates[:, 2]
        trivial_ev[3] = np.stack([np.zeros_like(x), -z,              y            ], axis=1).ravel()
        trivial_ev[4] = np.stack([z,                np.zeros_like(y), -x           ], axis=1).ravel()
        trivial_ev[5] = np.stack([-y,               x,               np.zeros_like(z)], axis=1).ravel()

        # SVD: singular values are ordered descending, so degenerate modes are always last
        U, S, _ = np.linalg.svd(trivial_ev.T, full_matrices=False)
        rank = np.sum(S > 1e-8)   # 5 for linear, 6 for non-linear

        if rank < 6:
            # TODO: log a warning here
            pass

        return U[:, :rank]