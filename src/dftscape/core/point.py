from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ..common import SerialisableDataClass
from ..common.distance import count_fragments
from .interfaces import Backend
from .symmetry_handlers import SymmetryRegistry


@dataclass(frozen=True)
class CriticalPoint(SerialisableDataClass):

    index: int
    energy: float
    xyz: str
    state: np.ndarray  # shape (d,)
    gradient: np.ndarray  # shape (d,)
    hessian: np.ndarray  # shape (d, d)
    eigenvalues: np.ndarray  # shape (d,)
    eigenvectors: np.ndarray  # shape (d, d)
    num_fragments: int = 1
    comment: str = ""

    @classmethod
    def from_state(
        cls, state: np.ndarray, backend: Backend, index: Optional[int] = None
    ) -> "CriticalPoint":
        # Compute properties at this state
        energy = backend.compute_energy(state)
        gradient = backend.compute_gradient(state)
        hessian = backend.compute_hessian(state)
        xyz = backend.get_xyz(state)

        # num_fragments (for reporting) still uses the original vdW definition
        num_fragments = count_fragments(xyz, mode="vdw")

        
        # Normal single molecule (or more than two covalent fragments)
        sym_handler = SymmetryRegistry.create("molecular")

        # Get the orthonormal constraint subspace
        constraint_basis = sym_handler.get_constraint_subspace(state)

        # Build the projector onto the internal space
        I = np.eye(state.size)
        if constraint_basis.size > 0:          # protects against degenerate edge cases
            P_constraint = constraint_basis @ constraint_basis.T
            P_int = I - P_constraint
        else:
            P_int = I.copy()

        # Project the Hessian → forces the constrained eigenvalues to *exactly* 0.0
        projected_hessian = P_int @ hessian @ P_int

        # Compute eigenvalues / eigenvectors of the projected Hessian
        eigenvalues, eigenvectors = np.linalg.eigh(projected_hessian)

        # Compute Morse index if not provided by caller
        if index is None:
            index = np.sum(eigenvalues < -1e-6)

        return cls(
            index=index,
            energy=energy,
            xyz=xyz,
            state=state.copy(),           # avoid reference issues
            gradient=gradient,
            hessian=projected_hessian,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            num_fragments=num_fragments,
        )

    def __str__(self) -> str:
        lines = [
            f"CriticalPoint (Index: {self.index}):",
            f"  Energy: {self.energy:.6f}",
            f"  Gradient norm: {np.linalg.norm(self.gradient):.6e}",
            f"  Number of fragments: {self.num_fragments}"
        ]

        # Add eigenvalue information
        neg_eigenvals = np.sum(self.eigenvalues < -1e-6)
        lines.append(f"  Morse index: {neg_eigenvals}")
        lines.append(
            f"  Eigenvalues range: [{self.eigenvalues.min():.6f}, {self.eigenvalues.max():.6f}]"
        )

        # Add XYZ coordinates (truncated if too long)
        xyz_lines = self.xyz.split("\n")
        #if len(xyz_lines) > 5:
        #    lines.append("  XYZ coordinates: (truncated)")
        #    lines.extend(f"    {line}" for line in xyz_lines[:5])
        #    lines.append(f"    ... ({len(xyz_lines) - 5} more lines)")
        #else:
        #    lines.append("  XYZ coordinates:")
        #    lines.extend(f"    {line}" for line in xyz_lines)
        lines.append("  XYZ coordinates:")
        lines.extend(f"    {line}" for line in xyz_lines)
        if self.comment:
            lines.append(f"  Comment: {self.comment}")

        return "\n".join(lines)
