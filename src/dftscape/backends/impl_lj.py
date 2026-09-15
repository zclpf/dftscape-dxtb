from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from .registry import BackendRegistry
from ..core.interfaces import Backend

# Minimal Periodic Table for XYZ parsing (Symbol -> Atomic Number)
PERIODIC_TABLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}

# Inverse map for XYZ generation
ATOMIC_NUMBERS = {v: k for k, v in PERIODIC_TABLE.items()}


@BackendRegistry.register("lj")
class LennardJonesBackend(Backend):
    """
    Backend for Lennard-Jones clusters.

    All atoms are treated as identical LJ particles (the standard setup for LJ
    cluster benchmarks, e.g. Wales' global-optimization database), regardless
    of the element symbol used in the XYZ file. The XYZ format is reused purely
    as an I/O convenience (the task states the file contains H atoms).

    Units: coordinates (both the input XYZ and all `state` vectors) are
    assumed to already be expressed in units of sigma, i.e. reduced/dimensionless
    LJ units. No length-unit conversion is performed anywhere in this backend
    -- sigma is therefore fixed at 1.0 by construction (that is exactly what
    "coordinates in units of sigma" means). epsilon only rescales the energy
    (and gradient/Hessian) magnitude and never affects the geometry of any
    stationary point (minima, transition states, etc.); it defaults to 1.0
    but can be overridden via `options["epsilon"]` if physical energy units
    are desired.
    """

    def __init__(
        self,
        cmp_mthd: str,
        memory: str,
        num_threads: int,
        options: Dict[str, Any],
        molecule_xyz: str,
        molecule_directives: Dict[str, Any],
        use_temp_file: bool = False,
    ) -> None:
        self.cmp_mthd: str = cmp_mthd
        self.memory: str = memory
        self.num_threads: int = num_threads
        self.options: Dict[str, Any] = options.copy() if options else {}

        # sigma is fixed at 1.0: coordinates are assumed to already be given
        # in units of sigma, so no length scale is ever applied. epsilon only
        # rescales energy magnitude (never geometry) and is configurable.
        self.sigma: float = 1.0
        self.epsilon: float = float(self.options.get("epsilon", 1.0))

        self.molecule_directives = molecule_directives

        # Parse Molecule XYZ to Numbers and Positions (positions in units of sigma)
        self.numbers, self.positions = self._parse_xyz(molecule_xyz)
        self.n_atoms: int = self.numbers.shape[0]

    def _parse_xyz(self, xyz: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parses XYZ string to atomic numbers and positions (units of sigma, unchanged)."""
        lines = xyz.strip().splitlines()
        if len(lines) < 3:
            raise ValueError("XYZ string is too short.")

        numbers_list = []
        coords_list = []

        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            symbol = parts[0]
            if symbol not in PERIODIC_TABLE:
                import re
                match = re.match(r"([A-Za-z]+)", symbol)
                if match and match.group(1) in PERIODIC_TABLE:
                    symbol = match.group(1)
                else:
                    raise ValueError(f"Unknown atom symbol: {symbol}")

            numbers_list.append(PERIODIC_TABLE[symbol])
            coords = [float(x) for x in parts[1:4]]
            coords_list.append(coords)

        numbers = np.array(numbers_list, dtype=int)
        positions = np.array(coords_list, dtype=float)
        return numbers, positions

    def __del__(self) -> None:
        pass

    # ---------------------------- LJ core (analytic) ----------------------------

    @staticmethod
    def _pairwise(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (diff, r):
          diff[i, j] = pos[i] - pos[j]   (N, N, 3)
          r[i, j]    = |diff[i, j]|      (N, N), with r[i, i] = inf to avoid
                       self-interaction / division-by-zero.
        """
        diff = pos[:, None, :] - pos[None, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        return diff, r

    def _energy_analytic(self, pos: np.ndarray) -> float:
        """E = 4*eps * sum_{i<j} [ (sigma/r_ij)^12 - (sigma/r_ij)^6 ]"""
        _, r = self._pairwise(pos)
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 * sr6
        # each unordered pair is counted twice in the full (N,N) sum
        e = 4.0 * self.epsilon * np.sum(sr12 - sr6) / 2.0
        return float(e)

    def _gradient_analytic(self, pos: np.ndarray) -> np.ndarray:
        """
        Analytic gradient dE/dr_i, flattened to shape (3N,).

        u(r)   = 4*eps*[(sigma/r)^12 - (sigma/r)^6]
        u'(r)  = (24*eps/r) * [(sigma/r)^6 - 2*(sigma/r)^12]
        dE/dr_i = sum_{j != i} u'(r_ij) * n_ij ,   n_ij = (r_i - r_j)/r_ij
        """
        diff, r = self._pairwise(pos)
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 * sr6

        # coeff = u'(r)/r, so that grad contribution = coeff * diff
        coeff = 24.0 * self.epsilon * (sr6 - 2.0 * sr12) / (r * r)
        coeff[np.isinf(r)] = 0.0  # zero out the (masked) diagonal

        grad = np.sum(coeff[:, :, None] * diff, axis=1)
        return grad.ravel()

    def _hessian_analytic(self, pos: np.ndarray) -> np.ndarray:
        """
        Analytic Hessian of the LJ potential, shape (3N, 3N).

        For a pairwise central potential u(r), the standard result is:
          H_ij (i != j) = -[ u''(r) (n_ij (x) n_ij) + (u'(r)/r) (I - n_ij (x) n_ij) ]
          H_ii           = -sum_{j != i} H_ij           (translational invariance)

        with u'(r), u''(r) the first/second derivatives of the LJ pair
        potential and n_ij = (r_i - r_j)/r_ij the unit separation vector.
        """
        n = self.n_atoms
        diff, r = self._pairwise(pos)

        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 * sr6

        u_prime = (24.0 * self.epsilon / r) * (sr6 - 2.0 * sr12)
        u_double_prime = (4.0 * self.epsilon / (r * r)) * (156.0 * sr12 - 42.0 * sr6)

        finite_mask = ~np.isinf(r)

        with np.errstate(invalid="ignore"):
            n_ij = diff / r[:, :, None]
        n_ij[~finite_mask] = 0.0

        u_prime = np.where(finite_mask, u_prime, 0.0)
        u_double_prime = np.where(finite_mask, u_double_prime, 0.0)
        r_safe = np.where(finite_mask, r, 1.0)

        eye3 = np.eye(3)
        outer = n_ij[:, :, :, None] * n_ij[:, :, None, :]

        coeff_par = u_double_prime
        coeff_perp = np.where(finite_mask, u_prime / r_safe, 0.0)

        Hij = -(
            coeff_par[:, :, None, None] * outer
            + coeff_perp[:, :, None, None] * (eye3[None, None, :, :] - outer)
        )
        idx = np.arange(n)
        Hij[idx, idx] = 0.0  # clear the masked diagonal blocks

        H = Hij.copy()
        H[idx, idx] = -np.sum(Hij, axis=1)  # translational-invariance diagonal blocks

        # (N, N, 3, 3) -> (3N, 3N)
        H_full = np.transpose(H, (0, 2, 1, 3)).reshape(3 * n, 3 * n)

        # symmetrize defensively against floating-point asymmetry
        return (H_full + H_full.T) / 2.0

    # ---------------------------- compute ----------------------------

    def compute_energy(self, state: np.ndarray) -> float:
        self._set_state(state)
        return self._energy_analytic(self.positions)

    def compute_gradient(self, state: np.ndarray) -> np.ndarray:
        self._set_state(state)
        return self._gradient_analytic(self.positions)

    def compute_hessian(self, state: np.ndarray) -> np.ndarray:
        self._set_state(state)
        return self._hessian_analytic(self.positions)

    # ---------------------------- state ----------------------------

    def _set_state(self, state: np.ndarray) -> None:
        coords = np.asarray(state, dtype=float).reshape(-1, 3)
        if coords.shape[0] != self.numbers.shape[0]:
            raise ValueError(
                f"State has {coords.shape[0]} atoms but molecule has {self.numbers.shape[0]}."
            )
        self.positions = coords

    # ---------------------------- I/O ----------------------------

    def get_xyz(self, state: np.ndarray) -> str:
        # state is already in units of sigma -- no conversion needed
        coords = np.asarray(state, dtype=float).reshape(-1, 3)

        natoms = len(self.numbers)
        lines = [f"{natoms}", "Generated by LennardJonesBackend"]

        for i in range(natoms):
            z = self.numbers[i]
            sym = ATOMIC_NUMBERS.get(z, "X")
            x, y, z_coord = coords[i]
            lines.append(f"{sym: <4} {x: >14.8f} {y: >14.8f} {z_coord: >14.8f}")

        return "\n".join(lines)

    def get_state_from_xyz(self, xyz: str) -> np.ndarray:
        lines = xyz.strip().splitlines()
        if len(lines) < 3:
            raise ValueError("XYZ string is too short.")
        coords_list = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ line: '{line}'")
            try:
                x, y, z = map(float, parts[1:4])
                coords_list.extend([x, y, z])
            except ValueError:
                raise ValueError(f"Invalid coordinate values in line: '{line}'")

        return np.array(coords_list)

    def optimize(self, state: np.ndarray) -> np.ndarray:
        """
        Optimizes geometry using SciPy's L-BFGS-B, driven by the analytic
        gradient (no autograd/finite-differences needed).
        """
        self._set_state(state)
        x0 = self.positions.ravel().copy()

        def fun(x: np.ndarray) -> float:
            pos = x.reshape(-1, 3)
            return self._energy_analytic(pos)

        def jac(x: np.ndarray) -> np.ndarray:
            pos = x.reshape(-1, 3)
            return self._gradient_analytic(pos)

        result = minimize(
            fun,
            x0,
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": 2000,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
        )

        self.positions = result.x.reshape(-1, 3)
        return self.positions.ravel()