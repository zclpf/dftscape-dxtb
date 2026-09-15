from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from ..common import SerialisableDataClass
from .point import CriticalPoint


# ------------------------------------------------------------------ #
#              Run conditions (to be stored in a stack)              #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class RunCondition(SerialisableDataClass):

    index: int  # = k
    state: np.ndarray  # (d,)
    perturbation_idx: int
    perturbation_strength: float
    perturbation_direction: np.ndarray  # (d,)
    subspace: np.ndarray  # (d, k)

    def get_perturbed_state(self) -> np.ndarray:
        return self.state + self.perturbation_strength * self.perturbation_direction

    def __str__(self) -> str:
        lines = [
            f"RunCondition (Target: {self.index}):",
            f"  Perturbation index: {self.perturbation_idx}",
            f"  Perturbation strength: {self.perturbation_strength:.6f}",
            f"  State shape: {self.state.shape}",
            f"  Perturbation direction shape: {self.perturbation_direction.shape}",
            f"  Subspace shape: {self.subspace.shape if self.subspace is not None else 'None'}",
        ]

        # Add state information (truncated if too long)
        if len(self.state) > 6:
            lines.append(
                f"  State: [{self.state[0]:.3f}, {self.state[1]:.3f}, {self.state[2]:.3f}, ..., {self.state[-1]:.3f}]"
            )
        else:
            lines.append(f"  State: {self.state}")

        # Add perturbation direction (truncated if too long)
        if len(self.perturbation_direction) > 6:
            lines.append(
                f"  Perturbation direction: [{self.perturbation_direction[0]:.3f}, {self.perturbation_direction[1]:.3f}, {self.perturbation_direction[2]:.3f}, ..., {self.perturbation_direction[-1]:.3f}]"
            )
        else:
            lines.append(f"  Perturbation direction: {self.perturbation_direction}")

        lines.append(f"  Perturbed state: {self.get_perturbed_state()}")

        return "\n".join(lines)


class RunConditionBuilder:

    @classmethod
    def build_up(
        cls,
        crit_point: CriticalPoint,
        perturbation_idx: int,
        perturbation_strength: float,
    ) -> RunCondition:
        if perturbation_idx < crit_point.index:
            raise ValueError(
                f"Perturbation index {perturbation_idx} must be >= critical point index {crit_point.index}."
            )

        index: int = crit_point.index + 1
        direction_idx: List[int] = list(range(crit_point.index)) + [perturbation_idx]
        # direction_idx: List[int] = list(range(index))  # ? Which is correct?
        return cls._build_run_condition(
            crit_point,
            index,
            direction_idx,
            perturbation_idx,
            perturbation_strength
        )
                
    @classmethod
    def build_down(
        cls,
        crit_point: CriticalPoint,
        perturbation_idx: int,
        perturbation_strength: float,
    ) -> RunCondition:
        if perturbation_idx >= crit_point.index:
            raise ValueError(
                f"Perturbation index {perturbation_idx} must be < critical point index {crit_point.index}."
            )

        index: int = crit_point.index - 1
        direction_idx: List[int] = [
            j for j in range(crit_point.index) if j != perturbation_idx
        ]

        return cls._build_run_condition(
            crit_point,
            index,
            direction_idx,
            perturbation_idx,
            perturbation_strength
        )

    @classmethod
    def build_up_all(
        cls,
        crit_point: CriticalPoint,
        perturbation_mag: float,
    ) -> List[List[RunCondition]]:
        run_conditions: List[List[RunCondition]] = []
        for perturbation_idx in range(crit_point.index, crit_point.eigenvalues.size):
            rc_plus: RunCondition = cls.build_up(
                crit_point,
                perturbation_idx,
                +perturbation_mag
            )
            rc_minus: RunCondition = cls.build_up(
                crit_point,
                perturbation_idx,
                -perturbation_mag
            )
            run_conditions.append([rc_plus, rc_minus])
        return run_conditions

    @classmethod
    def build_down_all(
        cls,
        crit_point: CriticalPoint,
        perturbation_mag: float,
    ) -> List[List[RunCondition]]:
        run_conditions: List[List[RunCondition]] = []
        for perturbation_idx in range(0, crit_point.index):
            rc_plus: RunCondition = cls.build_down(
                crit_point,
                perturbation_idx,
                perturbation_mag
            )
            rc_minus: RunCondition = cls.build_down(
                crit_point,
                perturbation_idx,
                -perturbation_mag
            )
            run_conditions.append([rc_plus, rc_minus])
        return run_conditions

    @classmethod
    def _build_run_condition(
        cls,
        crit_point: CriticalPoint,
        index: int,
        direction_idx: List[int],
        perturbation_idx: int,
        perturbation_strength: float
    ) -> RunCondition:
        subspace: np.ndarray = crit_point.eigenvectors[:, direction_idx]
        perturbation_direction: np.ndarray = crit_point.eigenvectors[
            :, perturbation_idx
        ]

        return RunCondition(
            index=index,
            perturbation_idx=perturbation_idx,
            perturbation_strength=perturbation_strength,
            perturbation_direction=perturbation_direction,
            state=crit_point.state.copy(),
            subspace=subspace
        )
