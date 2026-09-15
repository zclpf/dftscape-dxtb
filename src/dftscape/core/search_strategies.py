from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Union
import numpy as np
import logging
from .search_primitives import RunCondition
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .point import CriticalPoint
    from .search_primitives import RunCondition


class SearchStrategy(ABC):

    @abstractmethod
    def spawn_jobs(self, critical_point: "CriticalPoint") -> List["RunCondition"]:
        ...


class ExhaustiveUpwardSearchStrategy(SearchStrategy):

    def __init__(
        self, perturbation_magnitude: float = 0.1, eigen_threshold: float = 1e-6
    ):
        self.perturbation_magnitude = perturbation_magnitude
        self.eigen_threshold = eigen_threshold

    def spawn_jobs(self, critical_point: "CriticalPoint") -> List["RunCondition"]:
        from .search_primitives import RunConditionBuilder

        try:
            job_pairs = RunConditionBuilder.build_up_all(
                critical_point, self.perturbation_magnitude
            )
            # Flatten and return RunCondition list
            jobs: List["RunCondition"] = []
            for pair in job_pairs:
                for rc in pair:
                    jobs.append(rc)

            return jobs
        except Exception as e:
            logger.error(f"ExhaustiveUpward spawn_jobs failed: {e}")
            return []
            

class FirstNontrivialUpwardSearchStrategy(SearchStrategy):

    def __init__(
        self, perturbation_magnitude: float = 0.1, eigen_threshold: float = 1e-6
    ):
        self.perturbation_magnitude = perturbation_magnitude
        self.eigen_threshold = eigen_threshold

    def spawn_jobs(self, critical_point: "CriticalPoint") -> List["RunCondition"]:
        from .search_primitives import RunConditionBuilder

        # Need at least one available upward direction: eigenvalues array must
        # have size > critical_point.index (so we can increase index by 1).
        if critical_point.eigenvalues.size <= critical_point.index:
            return []

        # Choose the smallest eigenvalue index j >= critical_point.index
        # such that eigenvalue > eigen_threshold.
        eigs = critical_point.eigenvalues
        start_idx = critical_point.index
        perturbation_idx = None
        for j in range(start_idx, eigs.size):
            if eigs[j] > self.eigen_threshold:
                perturbation_idx = j
                break

        if perturbation_idx is None:
            return []

        rc_plus = RunConditionBuilder.build_up(
            critical_point,
            perturbation_idx=perturbation_idx,
            perturbation_strength=self.perturbation_magnitude
        )

        rc_minus = RunConditionBuilder.build_up(
            critical_point,
            perturbation_idx=perturbation_idx,
            perturbation_strength=-self.perturbation_magnitude
        )

        return [rc_plus, rc_minus]
        
        
class TargetedMultiModeUpwardSearchStrategy(SearchStrategy):
    """
    Spawns an upward search targeting one or multiple specific eigenvectors.
    If a list of modes is provided, the search index increases by the size of the list,
    effectively searching for a higher-order saddle point.
    """
    def __init__(
        self, target_modes: Union[int, List[int]], perturbation_magnitude: float = 0.1, eigen_threshold: float = 1e-6
    ):
        # Normalize input to a list for consistent handling
        if isinstance(target_modes, int):
            self.target_modes = [target_modes]
        else:
            self.target_modes = target_modes
            
        self.perturbation_magnitude = perturbation_magnitude
        self.eigen_threshold = eigen_threshold

    def spawn_jobs(self, critical_point: "CriticalPoint") -> List["RunCondition"]:
        eigs = critical_point.eigenvalues
        highest_mode_idx = max(self.target_modes)

        if highest_mode_idx >= eigs.size:
            logger.warning(f"Target mode {highest_mode_idx} exceeds available eigenvalues.")
            return []

        # Select exactly the eigenvector columns for the requested modes; the
        # ascent subspace handed to the dynamics contains only these.
        ascent_idx: List[int] = []
        cur_mode = 0
        remaining = len(self.target_modes)
        perturb_dir = np.zeros_like(critical_point.state)
        for j in range(eigs.size):
            if abs(eigs[j]) > self.eigen_threshold:
                cur_mode += 1
                if cur_mode in self.target_modes:
                    ascent_idx.append(j)
                    perturb_dir += critical_point.eigenvectors[:, j]
                    remaining -= 1
                if remaining == 0:
                    break

        perturb_dir = perturb_dir / np.linalg.norm(perturb_dir)
        ascent_subspace = critical_point.eigenvectors[:, ascent_idx]
        new_index = len(ascent_idx)

        rc_plus = RunCondition(
            index=new_index,
            perturbation_idx=ascent_idx[0],
            perturbation_strength=self.perturbation_magnitude,
            perturbation_direction=perturb_dir,
            state=critical_point.state.copy(),
            subspace=ascent_subspace
        )

        rc_minus = RunCondition(
            index=new_index,
            perturbation_idx=ascent_idx[0],
            perturbation_strength=-self.perturbation_magnitude,
            perturbation_direction=perturb_dir,
            state=critical_point.state.copy(),
            subspace=ascent_subspace
        )

        return [rc_plus, rc_minus]


class ExhaustiveDownwardSearchStrategy(SearchStrategy):
    """
    Spawns downward searches by perturbing along each unstable mode
    individually. For each of the k unstable modes, perturbs along that mode
    (in +/- directions) and follows (k-1)-MHiSD with the remaining k-1
    unstable modes as the ascent subspace.
    """
    def __init__(self, perturbation_magnitude: float = 0.1):
        self.perturbation_magnitude = perturbation_magnitude

    def spawn_jobs(self, critical_point: "CriticalPoint") -> List["RunCondition"]:
        from .search_primitives import RunConditionBuilder

        if critical_point.index == 0:
            return []

        try:
            # For each unstable mode: perturb along it and ascend the remaining
            # k-1 unstable modes. build_down_all returns +/- pairs.
            job_pairs = RunConditionBuilder.build_down_all(
                critical_point, self.perturbation_magnitude
            )
            jobs: List["RunCondition"] = []
            for pair in job_pairs:
                for rc in pair:
                    jobs.append(rc)

            return jobs

        except Exception as e:
            logger.error(f"ExhaustiveDownward spawn_jobs failed: {e}")
            return []
