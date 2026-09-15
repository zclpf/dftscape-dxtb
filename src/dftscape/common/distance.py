from __future__ import annotations
from typing import Any, cast, Protocol
import numpy as np
import itertools
import logging
from rmsd.calculate_rmsd import (
    get_coordinates_xyz_lines,
    kabsch_rmsd,
    quaternion_rmsd,
    rmsd,
)

from collections import Counter
from itertools import permutations as iter_permutations, product as iter_product
from scipy.optimize import linear_sum_assignment
from .registry import BaseRegistry

# Optional RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# covalent radii
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    'Br': 1.20, 'I': 1.39, 'B': 0.85
}
COVALENT_TOLERANCE = 1.2

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75,
    'Br': 1.85, 'I': 1.98, 'B': 1.92
}
VDW_TOLERANCE = 1.2


def count_fragments(xyz_string, return_indices=False, mode="covalent"):
    atom_symbols = []
    coords = []

    for line in xyz_string.strip().split('\n'):
        parts = line.split()
        if len(parts) == 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atom_symbols.append(parts[0].capitalize())
                coords.append([x, y, z])
            except ValueError:
                continue

    if not coords:
        return (0, []) if return_indices else 0

    coords = np.array(coords)
    num_atoms = len(coords)

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    if mode == "covalent":
        radii = np.array([COVALENT_RADII.get(sym, 0.76) for sym in atom_symbols])
        cutoff_matrix = COVALENT_TOLERANCE * (radii[:, np.newaxis] + radii[np.newaxis, :])
    elif mode == "vdw":
        radii = np.array([VDW_RADII.get(sym, 1.7) for sym in atom_symbols])
        cutoff_matrix = VDW_TOLERANCE * (radii[:, np.newaxis] + radii[np.newaxis, :])

    adjacency = dist_matrix < cutoff_matrix
    visited = np.zeros(num_atoms, dtype=bool)
    fragments = []

    for i in range(num_atoms):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            current_fragment = [i]
            while queue:
                current = queue.pop(0)
                neighbors = np.where(adjacency[current] & ~visited)[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    current_fragment.append(neighbor)
            fragments.append(sorted(current_fragment))

    if return_indices:
        return len(fragments), fragments
    return len(fragments)


def _hungarian_permute(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Return coords2 reordered to minimise the sum of squared distances to coords1.

    Assumes coords1 and coords2 are already centred and have the same atom ordering
    (same element at each position). This is called once per same-element group inside
    a fragment, so the cost matrix is always small.
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(
        coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :], axis=-1
    )  # shape (n, n)
    row_idx, col_idx = linear_sum_assignment(cost)
    # col_idx[i] is the index in coords2 that should pair with row i of coords1
    perm = np.empty(len(coords2), dtype=int)
    perm[row_idx] = col_idx
    return coords2[perm]


def _align_fragment_pair(
    frag_c1: np.ndarray,
    frag_c2: np.ndarray,
    atoms1: list[str],
    atoms2: list[str],
) -> float:
    """Compute Kabsch RMSD between two same-sized fragments with Hungarian
    pre-permutation per element type.

    This replaces the old call to rmsd_func(..., reorder_method=...) which
    silently had no effect because kabsch_rmsd / quaternion_rmsd / rmsd do
    not accept that parameter.
    """
    from collections import defaultdict

    if len(atoms1) != len(atoms2):
        return float('inf')

    # Centre both fragments
    c1 = frag_c1 - frag_c1.mean(axis=0)
    c2 = frag_c2 - frag_c2.mean(axis=0)

    # Build a global permutation map initialised to identity
    perm_map = list(range(len(atoms1)))

    # Group indices by element
    groups1: dict[str, list[int]] = defaultdict(list)
    groups2: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(atoms1):
        groups1[a].append(i)
    for i, a in enumerate(atoms2):
        groups2[a].append(i)

    # Check element composition matches
    if set(groups1.keys()) != set(groups2.keys()):
        return float('inf')
    for el in groups1:
        if len(groups1[el]) != len(groups2[el]):
            return float('inf')

    # Hungarian assignment per element group
    for el in groups1:
        idx1 = groups1[el]
        idx2 = groups2[el]
        if len(idx1) == 1:
            perm_map[idx1[0]] = idx2[0]
            continue
        g1 = c1[idx1]          # (m, 3)
        g2 = c2[idx2]          # (m, 3)
        g2_perm = _hungarian_permute(g1, g2)
        # g2_perm[i] should pair with g1[i]; write back into the global perm
        cost = np.linalg.norm(
            g1[:, np.newaxis, :] - g2[np.newaxis, :, :], axis=-1
        )
        from scipy.optimize import linear_sum_assignment
        _, col = linear_sum_assignment(cost)
        for local_i, local_j in enumerate(col):
            perm_map[idx1[local_i]] = idx2[local_j]

    c2_perm = c2[perm_map]

    # Kabsch rotation
    H = c2_perm.T @ c1
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    c2_rot = c2_perm @ R.T
    return float(np.sqrt(np.mean((c1 - c2_rot) ** 2)))


class DistanceFunction(Protocol):
    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        ...


class DistanceFunctionBase:
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__


class DistanceRegistry(BaseRegistry):
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> DistanceFunction:
        return cast(DistanceFunction, super().create(name, *args, **kwargs))


@DistanceRegistry.register("rmsd")
class RMSDDistance(DistanceFunctionBase):

    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        if not isinstance(xyz1, str):
            xyz1 = xyz1.xyz
        if not isinstance(xyz2, str):
            xyz2 = xyz2.xyz
        return RMSDDistance.compute_rmsd(xyz1, xyz2, **kwargs)

    @staticmethod
    def compute_rmsd(
        xyz1: str,
        xyz2: str,
        method: str = "kabsch",
        translate: bool = True,
        **kwargs: Any,
    ) -> float:
        import itertools

        lines1 = xyz1.strip().split("\n")
        lines2 = xyz2.strip().split("\n")

        atoms1, coords1 = get_coordinates_xyz_lines(lines1)
        atoms2, coords2 = get_coordinates_xyz_lines(lines2)

        if len(atoms1) != len(atoms2):
            raise ValueError(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}")

        # Select the whole-molecule alignment function (used only for intact molecules)
        if method == "kabsch":
            rmsd_func = kabsch_rmsd
        elif method == "quaternion":
            rmsd_func = quaternion_rmsd
        elif method == "none":
            rmsd_func = rmsd
        else:
            raise ValueError(f"Unknown method: {method}")

        # --- Fragment extraction ---
        n_frags1, frags1_indices = count_fragments(xyz1, return_indices=True, mode='covalent')
        n_frags2, frags2_indices = count_fragments(xyz2, return_indices=True, mode='covalent')

        if n_frags1 != n_frags2:
            return float('inf')

        # --- Fragmented case ---
        if n_frags1 > 1:
            total_atoms = len(atoms1)
            best_overall = float('inf')

            for frags2_perm in itertools.permutations(frags2_indices):

                # Quick size-compatibility check before doing any real work
                if any(
                    len(frags1_indices[i]) != len(frags2_perm[i])
                    for i in range(n_frags1)
                ):
                    continue

                weighted_sum = 0.0
                valid = True

                for i in range(n_frags1):
                    idx1 = frags1_indices[i]
                    idx2 = frags2_perm[i]

                    frag_atoms1 = [atoms1[j] for j in idx1]
                    frag_atoms2 = [atoms2[j] for j in idx2]

                    frag_c1 = coords1[idx1]
                    frag_c2 = coords2[idx2]

                    try:
                        # FIX: use our own aligned RMSD instead of passing
                        # reorder_method to kabsch_rmsd (which doesn't accept it).
                        frag_rmsd_val = _align_fragment_pair(
                            frag_c1, frag_c2, frag_atoms1, frag_atoms2
                        )
                    except Exception:
                        valid = False
                        break

                    if frag_rmsd_val == float('inf'):
                        valid = False
                        break

                    # FIX: size-weighted sum instead of max, so large and small
                    # fragments contribute proportionally to the overall distance.
                    weighted_sum += frag_rmsd_val * len(idx1)

                if valid:
                    candidate = weighted_sum / total_atoms
                    best_overall = min(best_overall, candidate)

            return best_overall

        # --- Intact molecule: delegate to rmsd library as before ---
        return rmsd_func(
            coords1,
            coords2,
            translate=translate,
            **kwargs,
        )

@DistanceRegistry.register("euclidean")
class EuclideanDistance(DistanceFunctionBase):

    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        if not isinstance(xyz1, str):
            xyz1 = xyz1.xyz
        if not isinstance(xyz2, str):
            xyz2 = xyz2.xyz
        lines1 = xyz1.strip().split("\n")
        lines2 = xyz2.strip().split("\n")
        atoms1, coords1 = get_coordinates_xyz_lines(lines1)
        atoms2, coords2 = get_coordinates_xyz_lines(lines2)
        if len(atoms1) != len(atoms2):
            raise ValueError(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}")
        align = kwargs.get("align", True)
        if align:
            coords1 = coords1 - np.mean(coords1, axis=0)
            coords2 = coords2 - np.mean(coords2, axis=0)
        return float(np.linalg.norm(coords1 - coords2))


@DistanceRegistry.register("max_coordinate")
class MaxCoordinateDistance(DistanceFunctionBase):

    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        if not isinstance(xyz1, str):
            xyz1 = xyz1.xyz
        if not isinstance(xyz2, str):
            xyz2 = xyz2.xyz
        lines1 = xyz1.strip().split("\n")
        lines2 = xyz2.strip().split("\n")
        atoms1, coords1 = get_coordinates_xyz_lines(lines1)
        atoms2, coords2 = get_coordinates_xyz_lines(lines2)
        if len(atoms1) != len(atoms2):
            raise ValueError(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}")
        return float(np.max(np.linalg.norm(coords1 - coords2, axis=1)))


@DistanceRegistry.register("rmsd_manual")
class ManualRMSD(DistanceFunctionBase):
    # Unchanged — kept verbatim
    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        permutation_method = kwargs.get("permutation_method", "exhaustive")
        max_exhaustive_atoms = kwargs.get("max_exhaustive_atoms", 8)
        if not isinstance(xyz1, str):
            xyz1 = xyz1.xyz
        if not isinstance(xyz2, str):
            xyz2 = xyz2.xyz
        return self.compute_optimal_rmsd(xyz1, xyz2, permutation_method, max_exhaustive_atoms)

    def compute_optimal_rmsd(self, xyz1, xyz2, permutation_method="hungarian", max_exhaustive_atoms=6):
        atoms1, coords1 = self._parse_xyz(xyz1)
        atoms2, coords2 = self._parse_xyz(xyz2)
        if len(atoms1) != len(atoms2):
            raise ValueError(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}")
        return self._compute_with_permutation(atoms1, coords1, atoms2, coords2, permutation_method, max_exhaustive_atoms)

    def _parse_xyz(self, xyz_string):
        lines = xyz_string.strip().split("\n")
        n_atoms = int(lines[0])
        atoms, coords = [], []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return atoms, np.array(coords)

    def _compute_with_permutation(self, atoms1, coords1, atoms2, coords2, permutation_method, max_exhaustive_atoms):
        from collections import defaultdict
        groups1, groups2 = defaultdict(list), defaultdict(list)
        for i, a in enumerate(atoms1): groups1[a].append(i)
        for i, a in enumerate(atoms2): groups2[a].append(i)
        for atom_type in groups1:
            if len(groups1[atom_type]) != len(groups2.get(atom_type, [])):
                raise ValueError(f"Atom type {atom_type} count mismatch")
        if permutation_method == "none":
            return self._kabsch_rmsd(coords1, coords2)
        elif permutation_method == "exhaustive":
            return self._exhaustive_permutation_rmsd(groups1, groups2, coords1, coords2, max_exhaustive_atoms)
        elif permutation_method == "hungarian":
            return self._hungarian_permutation_rmsd(groups1, groups2, coords1, coords2)
        elif permutation_method == "greedy":
            return self._greedy_permutation_rmsd(groups1, groups2, coords1, coords2)
        else:
            raise ValueError(f"Unknown permutation method: {permutation_method}")

    def _exhaustive_permutation_rmsd(self, groups1, groups2, coords1, coords2, max_exhaustive_atoms):
        from itertools import permutations, product
        permutation_options = []
        for atom_type in groups1:
            indices1 = groups1[atom_type]
            if len(indices1) > max_exhaustive_atoms:
                return self._greedy_permutation_rmsd(groups1, groups2, coords1, coords2)
            permutation_options.append([indices1] if len(indices1) <= 1 else list(permutations(indices1)))
        best_rmsd = float("inf")
        for perm_combo in product(*permutation_options):
            perm_map = list(range(len(coords1)))
            combo_idx = 0
            for atom_type in groups1:
                indices1 = groups1[atom_type]
                if len(indices1) > 1:
                    for orig_idx, new_idx in zip(indices1, perm_combo[combo_idx]):
                        perm_map[orig_idx] = new_idx
                    combo_idx += 1
            best_rmsd = min(best_rmsd, self._kabsch_rmsd(coords1, coords2[perm_map]))
        return best_rmsd

    def _hungarian_permutation_rmsd(self, groups1, groups2, coords1, coords2):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return self._greedy_permutation_rmsd(groups1, groups2, coords1, coords2)
        best_rmsd = float("inf")
        current_coords2 = coords2.copy()
        prev_rmsd = None
        for iteration in range(5):
            coords1_c = coords1 - np.mean(coords1, axis=0)
            coords2_c = current_coords2 - np.mean(current_coords2, axis=0)
            perm_map = list(range(len(coords1)))
            for atom_type in groups1:
                indices1, indices2 = groups1[atom_type], groups2[atom_type]
                if len(indices1) <= 1:
                    continue
                n = len(indices1)
                cost_matrix = np.zeros((n, n))
                for i, idx1 in enumerate(indices1):
                    for j, idx2 in enumerate(indices2):
                        cost_matrix[i, j] = np.linalg.norm(coords1_c[idx1] - coords2_c[idx2])
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                for i, j in zip(row_indices, col_indices):
                    perm_map[indices1[i]] = indices2[j]
            coords2_perm = coords2[perm_map]
            r = self._kabsch_rmsd(coords1, coords2_perm)
            if r < best_rmsd:
                best_rmsd = r
            coords1_c = coords1 - np.mean(coords1, axis=0)
            coords2_perm_c = coords2_perm - np.mean(coords2_perm, axis=0)
            H = coords2_perm_c.T @ coords1_c
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            current_coords2 = coords2_perm_c @ R.T + np.mean(coords1, axis=0)
            if prev_rmsd is not None and abs(r - prev_rmsd) < 1e-8:
                break
            prev_rmsd = r
        return best_rmsd

    def _greedy_permutation_rmsd(self, groups1, groups2, coords1, coords2):
        coords1_c = coords1 - np.mean(coords1, axis=0)
        coords2_c = coords2 - np.mean(coords2, axis=0)
        perm_map = list(range(len(coords1)))
        for atom_type in groups1:
            indices1 = groups1[atom_type]
            indices2 = groups2[atom_type].copy()
            if len(indices1) == 1:
                continue
            for idx1 in indices1:
                best_d, best_idx2, best_pos = float("inf"), None, None
                for pos, idx2 in enumerate(indices2):
                    d = np.linalg.norm(coords1_c[idx1] - coords2_c[idx2])
                    if d < best_d:
                        best_d, best_idx2, best_pos = d, idx2, pos
                perm_map[idx1] = best_idx2
                indices2.pop(best_pos)
        return self._kabsch_rmsd(coords1, coords2[perm_map])

    def _kabsch_rmsd(self, coords1, coords2):
        coords1_c = coords1 - np.mean(coords1, axis=0)
        coords2_c = coords2 - np.mean(coords2, axis=0)
        H = coords2_c.T @ coords1_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return float(np.sqrt(np.mean((coords1_c - coords2_c @ R.T) ** 2)))


def parse_xyz(xyz_str: str) -> tuple[list[str], np.ndarray]:
    """Robustly parses an XYZ string into atomic symbols and a coordinate array."""
    atoms, coords = [], []
    for line in xyz_str.strip().split('\n'):
        parts = line.split()
        # Look for lines with exactly 4 components: Symbol X Y Z
        if len(parts) == 4:
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                atoms.append(parts[0].capitalize())
            except ValueError:
                # If they aren't numbers (e.g., a 4-word comment line), skip it
                continue
    return atoms, np.array(coords)


def get_fragments(coords: np.ndarray, atoms: list[str], mode: str = "covalent") -> list[list[int]]:
    """Optimized fragment counter returning lists of indices."""
    num_atoms = len(coords)
    if num_atoms == 0:
        return []

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    if mode == "covalent":
        radii = np.array([COVALENT_RADII.get(sym, 0.76) for sym in atoms])
        cutoff_matrix = COVALENT_TOLERANCE * (radii[:, np.newaxis] + radii[np.newaxis, :])
    elif mode == "vdw":
        radii = np.array([VDW_RADII.get(sym, 1.7) for sym in atoms])
        cutoff_matrix = VDW_TOLERANCE * (radii[:, np.newaxis] + radii[np.newaxis, :])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    adjacency = dist_matrix < cutoff_matrix
    visited = np.zeros(num_atoms, dtype=bool)
    fragments = []

    for i in range(num_atoms):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            current_fragment = [i]
            while queue:
                current = queue.pop(0)
                neighbors = np.where(adjacency[current] & ~visited)[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    current_fragment.append(neighbor)
            fragments.append(sorted(current_fragment))

    return fragments


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (Q - cQ).T @ (P - cP)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    Vt[-1] *= d
    return Vt.T @ U.T, cP, cQ

def apply_transform(coords: np.ndarray, R: np.ndarray, centroid_target: np.ndarray, centroid_source: np.ndarray) -> np.ndarray:
    return (coords - centroid_source) @ R.T + centroid_target

def hungarian_match(c1: np.ndarray, c2: np.ndarray, a1: list[str], a2: list[str]) -> np.ndarray:
    n = len(c1)
    cost = np.full((n, n), 1e18)
    for i in range(n):
        for j in range(n):
            if a1[i] == a2[j]:
                diff = c1[i] - c2[j]
                cost[i, j] = float(diff @ diff)
    _, col = linear_sum_assignment(cost)
    return col

def compute_rmsd_vdw(
    xyz1: str,
    xyz2: str,
    *,
    max_iter: int = 50,
) -> float:
    """
    Minimum RMSD between two vdW complexes using pure NumPy arrays and XYZ strings.
    """
    atoms1, coords1 = parse_xyz(xyz1)
    atoms2, coords2 = parse_xyz(xyz2)

    if len(atoms1) != len(atoms2):
        raise ValueError(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}.")
    if Counter(atoms1) != Counter(atoms2):
        raise ValueError(f"Element composition mismatch.")

    # ── Fragment detection ────────────────────────────────────────────────────
    frags1 = get_fragments(coords1, atoms1, mode="covalent")
    frags2 = get_fragments(coords2, atoms2, mode="covalent")

    if len(frags1) != len(frags2):
        return float('inf')

    n_frags = len(frags1)
    
    # Create canonical composition signatures for each fragment (e.g., "C6H6", "Ar1")
    def get_comp_signature(frag_indices, atoms):
        comp = Counter(atoms[k] for k in frag_indices)
        return "".join(f"{k}{v}" for k, v in sorted(comp.items()))

    comps1 = [get_comp_signature(f, atoms1) for f in frags1]
    comps2 = [get_comp_signature(f, atoms2) for f in frags2]
    
    # ── Composition-Aware Permutation Generation ──────────────────────────────
    groups1 = {}
    for i, comp in enumerate(comps1):
        groups1.setdefault(comp, []).append(i)
        
    groups2 = {}
    for i, comp in enumerate(comps2):
        groups2.setdefault(comp, []).append(i)
        
    if groups1.keys() != groups2.keys():
        return float('inf')
        
    group_mappings = []
    for comp in groups1:
        idx1_list = groups1[comp]
        idx2_list = groups2[comp]
        if len(idx1_list) != len(idx2_list):
            return float('inf')
        
        perms_for_group = []
        for p in iter_permutations(idx2_list):
            perms_for_group.append(list(zip(idx1_list, p)))
        group_mappings.append(perms_for_group)

    anchor_i = max(range(n_frags), key=lambda i: len(frags1[i]))
    best_rmsd = float("inf")

    # Iterate over the Cartesian product of the valid group mappings
    for mapping_combination in iter_product(*group_mappings):
        
        frag_perm = [None] * n_frags
        for group_map in mapping_combination:
            for i1, i2 in group_map:
                frag_perm[i1] = i2
                
        frags2m = [frags2[j] for j in frag_perm]

        # ── Step 1: Hungarian-first anchor initialisation ─────────────────────
        idx1_anc = frags1[anchor_i]
        idx2_anc = frags2m[anchor_i]

        c1_anc_raw = coords1[idx1_anc]
        c2_anc_raw = coords2[idx2_anc]
        a1_anc = [atoms1[k] for k in idx1_anc]
        a2_anc = [atoms2[k] for k in idx2_anc]

        c1_anc_cent = c1_anc_raw - c1_anc_raw.mean(axis=0)
        c2_anc_cent = c2_anc_raw - c2_anc_raw.mean(axis=0)

        inner_anc = hungarian_match(c1_anc_cent, c2_anc_cent, a1_anc, a2_anc)
        idx2_anc_matched = [idx2_anc[lk] for lk in inner_anc]

        R_init, c1_anc, c2_anc = kabsch_align(c1_anc_raw, coords2[idx2_anc_matched])
        coords2_work = apply_transform(coords2, R_init, c1_anc, c2_anc)

        global_perm = np.arange(len(atoms1), dtype=int)
        for k, lk in enumerate(inner_anc):
            global_perm[idx1_anc[k]] = idx2_anc[lk]

        prev_perm: np.ndarray | None = None

        # ── Step 2: Iterative Kabsch–Hungarian ────────────────────────────────
        for _ in range(max_iter):
            new_perm = global_perm.copy()
            for i in range(n_frags):
                idx1 = frags1[i]
                idx2 = frags2m[i]
                inner = hungarian_match(
                    coords1[idx1], coords2_work[idx2],
                    [atoms1[k] for k in idx1],
                    [atoms2[k] for k in idx2],
                )
                for local_k, local_lk in enumerate(inner):
                    new_perm[idx1[local_k]] = idx2[local_lk]

            if prev_perm is not None and np.array_equal(new_perm, prev_perm):
                break
            
            prev_perm = new_perm.copy()
            global_perm = new_perm

            R_glob, c1_g, c2_g = kabsch_align(coords1, coords2[global_perm])
            coords2_work = apply_transform(coords2, R_glob, c1_g, c2_g)

        # ── Step 3: RMSD ──────────────────────────────────────────────────────
        coords2_final = coords2_work[global_perm]
        diff = coords1 - coords2_final
        rmsd_val = float(np.sqrt(np.mean(np.einsum("ij,ij->i", diff, diff))))
        best_rmsd = min(best_rmsd, rmsd_val)

    return best_rmsd


@DistanceRegistry.register("rmsd_smart")
class SmartRMSD(DistanceFunctionBase):
    """
    Unified RMSD router that dynamically selects the best alignment algorithm.
    
    Architecture:
    1. Covalent Intact Molecules -> RDKit Graph Isomorphism (Fast, strict)
    2. Fragmented/vdW Complexes -> Custom Iterative Spatial Alignment (Handles floppiness)
    3. Transition States/RDKit Failures -> Custom Spatial Alignment Fallback
    """

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        # 1. Normalize inputs to raw XYZ strings
        if not isinstance(xyz1, str):
            xyz1 = xyz1.xyz
        if not isinstance(xyz2, str):
            xyz2 = xyz2.xyz

        # 2. Fast parse to atoms and coordinates
        atoms1, coords1 = parse_xyz(xyz1)
        atoms2, coords2 = parse_xyz(xyz2)

        if len(atoms1) != len(atoms2):
            self.logger.debug("SmartRMSD: Rejecting due to atom count mismatch.")
            return float('inf')

        # 3. Determine Macroscopic and Chemical State
        frags1 = get_fragments(coords1, atoms1, mode="covalent")
        frags2 = get_fragments(coords2, atoms2, mode="covalent")
        
        n_covalent_1 = len(frags1)
        n_covalent_2 = len(frags2)

        # Topological Barrier: If they don't have the same covalent pieces, they are not the same node
        if n_covalent_1 != n_covalent_2:
            self.logger.debug(f"SmartRMSD: Rejecting. Covalent fragments differ ({n_covalent_1} vs {n_covalent_2}).")
            return float('inf')

        n_vdw_1 = len(get_fragments(coords1, atoms1, mode="vdw"))
        n_vdw_2 = len(get_fragments(coords2, atoms2, mode="vdw"))

        # Macroscopic Barrier: Do not merge a bound vdW complex with dissociated gas
        if n_vdw_1 != n_vdw_2:
            self.logger.debug(f"SmartRMSD: Rejecting. Macroscopic state differs (vdW fragments: {n_vdw_1} vs {n_vdw_2}).")
            return float('inf')

        # -----------------------------------------------------------------
        # ROUTE A: Fragmented System -> Custom Spatial Engine
        # -----------------------------------------------------------------
        if n_covalent_1 > 1:
            self.logger.debug("SmartRMSD: Routing to Custom Spatial Engine (Fragmented/vdW System)")
            try:
                return compute_rmsd_vdw(xyz1, xyz2, **kwargs)
            except ValueError as e:
                self.logger.error(f"Spatial RMSD failed: {e}")
                return float('inf')

        # -----------------------------------------------------------------
        # ROUTE B: Intact Covalent Molecule -> RDKit Graph Engine
        # -----------------------------------------------------------------
        self.logger.debug("SmartRMSD: Routing to RDKit Graph Engine (Intact Covalent Molecule)")
        
        if not RDKIT_AVAILABLE:
            self.logger.warning("RDKit missing. Falling back to Spatial Engine for intact molecule.")
            return compute_rmsd_vdw(xyz1, xyz2, **kwargs)

        try:
            m1 = Chem.MolFromXYZBlock(xyz1)
            m2 = Chem.MolFromXYZBlock(xyz2)

            if m1 is None or m2 is None:
                raise ValueError("RDKit failed to construct graph from XYZ.")

            rmsd_value = rdMolAlign.GetBestRMS(m1, m2, **kwargs)
            return float(rmsd_value)

        except Exception as e:
            # -----------------------------------------------------------------
            # FALLBACK: Transition State Rescue
            # If RDKit fails (usually due to weird valences in a TS), 
            # rescue the calculation with the spatial engine!
            # -----------------------------------------------------------------
            self.logger.warning(f"RDKit graph generation failed ({str(e)}). Likely a transition state. Falling back to Spatial Engine.")
            try:
                return compute_rmsd_vdw(xyz1, xyz2, **kwargs)
            except Exception as e_fallback:
                self.logger.error(f"Fallback Spatial RMSD also failed: {e_fallback}")
                return float('inf')


@DistanceRegistry.register("rmsd_rdkit")
class RDKitRMSD(DistanceFunctionBase):
 
    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for RDKitRMSD but is not installed.")
        
        if not isinstance(xyz1, str):
            xyz1 = xyz1.xyz
        if not isinstance(xyz2, str):
            xyz2 = xyz2.xyz

        # --- Helper to safely extract atoms and coordinates ---
        def _parse_xyz(xyz_str: str):
            atoms, coords = [], []
            for line in xyz_str.strip().split('\n'):
                parts = line.split()
                if len(parts) == 4:
                    try:
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        atoms.append(parts[0].capitalize())
                    except ValueError:
                        continue
            return atoms, np.array(coords)

        # --- Helper to build valid XYZ string blocks for RDKit fragments ---
        def _make_xyz_block(atoms: list, coords: np.ndarray) -> str:
            lines = [str(len(atoms)), "fragment"]
            for a, c in zip(atoms, coords):
                lines.append(f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
            return "\n".join(lines)

        atoms1, coords1 = _parse_xyz(xyz1)
        atoms2, coords2 = _parse_xyz(xyz2)

        if len(atoms1) != len(atoms2):
            raise ValueError(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}")

        # 1. Primary Chemical Topology Check (Covalent)
        n_frags1, frags1_indices = count_fragments(xyz1, return_indices=True, mode='covalent')
        n_frags2, frags2_indices = count_fragments(xyz2, return_indices=True, mode='covalent')

        if n_frags1 != n_frags2:
            return float('inf')

        # --- Fragmented Case ---
        if n_frags1 > 1:
            
            # -----------------------------------------------------------------
            # NEW: Secondary Macroscopic State Check (Van der Waals)
            # Prevents merging of bound vdW complexes with fully dissipated gas fragments
            # -----------------------------------------------------------------
            n_vdw1 = count_fragments(xyz1, return_indices=False, mode='vdw')
            n_vdw2 = count_fragments(xyz2, return_indices=False, mode='vdw')
            
            if n_vdw1 != n_vdw2:
                return float('inf')
            # -----------------------------------------------------------------

            total_atoms = len(atoms1)
            best_overall = float('inf')

            for frags2_perm in itertools.permutations(frags2_indices):
                
                # Quick size-compatibility check
                if any(len(frags1_indices[i]) != len(frags2_perm[i]) for i in range(n_frags1)):
                    continue

                sum_sq_dev = 0.0
                valid = True

                for i in range(n_frags1):
                    idx1 = frags1_indices[i]
                    idx2 = frags2_perm[i]

                    frag_atoms1 = [atoms1[j] for j in idx1]
                    frag_atoms2 = [atoms2[j] for j in idx2]
                    frag_c1 = coords1[idx1]
                    frag_c2 = coords2[idx2]

                    # Package the spatial fragments back into RDKit-readable strings
                    frag_xyz1 = _make_xyz_block(frag_atoms1, frag_c1)
                    frag_xyz2 = _make_xyz_block(frag_atoms2, frag_c2)

                    try:
                        m1_frag = Chem.MolFromXYZBlock(frag_xyz1)
                        m2_frag = Chem.MolFromXYZBlock(frag_xyz2)
                        
                        if m1_frag is None or m2_frag is None:
                            valid = False
                            break
                        
                        # RDKit handles the alignment and Hungarian atom matching natively
                        frag_rmsd_val = float(rdMolAlign.GetBestRMS(m1_frag, m2_frag, **kwargs))
                    except Exception:
                        valid = False
                        break

                    # Un-root the RMSD, multiply by N atoms to get total squared deviation
                    frag_sq_dev = (frag_rmsd_val ** 2) * len(idx1)
                    sum_sq_dev += frag_sq_dev

                if valid:
                    # Divide by total atoms, then take the final root
                    candidate = np.sqrt(sum_sq_dev / total_atoms)
                    best_overall = min(best_overall, float(candidate))

            return best_overall

        # --- Intact Molecule Case ---
        try:
            m1 = Chem.MolFromXYZBlock(xyz1)
            m2 = Chem.MolFromXYZBlock(xyz2)
            if m1 is None:
                raise ValueError("Failed to parse first molecular structure from XYZ")
            if m2 is None:
                raise ValueError("Failed to parse second molecular structure from XYZ")
            return float(rdMolAlign.GetBestRMS(m1, m2, **kwargs))
        except Exception as e:
            raise ValueError(f"Error calculating RDKit RMSD: {str(e)}") from e


@DistanceRegistry.register("energy")
class EnergyDistance(DistanceFunctionBase):

    def __call__(self, xyz1: str | Any, xyz2: str | Any, **kwargs: Any) -> float:
        if isinstance(xyz1, str):
            raise ValueError("Failed to parse first molecular")
        if isinstance(xyz2, str):
            raise ValueError("Failed to parse second molecular")
        return float(np.abs(xyz1.energy - xyz2.energy) + np.abs(xyz1.index - xyz2.index))
