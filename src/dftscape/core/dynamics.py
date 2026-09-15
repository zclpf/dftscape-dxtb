from __future__ import annotations
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from .interfaces import Backend
from .symmetry_handlers import SymmetryHandler
from .subspace_handlers import SubspaceHandler
from .point import CriticalPoint
from .search_primitives import RunCondition
from .metrics import Metric
from ..common.distance import count_fragments

import os
import csv
from datetime import datetime

import resource

class Dynamics(ABC):

    def __init__(self, backend: Backend, log_interval: Optional[int]=1, maxiter: int=10000, dt: float=0.5, grad_norm_tol: float=5e-06) -> None:
        self.backend: Backend = backend
        self.metrics: List[Metric] = []
        self.log_interval: int = log_interval
        self.maxiter: int = maxiter
        self.dt: float = dt
        self.grad_norm_tol: float = grad_norm_tol
        self.state: Optional[np.ndarray] = None
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _make_step(self, state: np.ndarray, step: int) -> np.ndarray:
        pass

    def _log(self, step: int) -> None:
        if self.log_interval is not None and step % self.log_interval == 0:
            metrics_values: List[float] = [metric.compute(self, step, self.dt) for metric in self.metrics]
            metrics_str: str = ' | '.join((f"{metric.name}: {value:{metric.format_str.lstrip(':')}}" for metric, value in zip(self.metrics, metrics_values)))
            self.logger.info('Step: %4d | %s', step, metrics_str)

    def _update(self, step: int) -> None:
        pass

    def _check_stopping_criterion(self, step: int) -> tuple[bool, bool]:
        return (False, False)

    @abstractmethod
    def run(self, run_condition: RunCondition) -> Optional[CriticalPoint]:
        pass

class HiSD(Dynamics):

    def __init__(
        self,
        backend: Backend,
        subspace_handler: SubspaceHandler,
        symmetry_handler: SymmetryHandler,
        subspace_update_interval: int = 1,
        log_interval: Optional[int] = 1,
        check_stop_interval: int = 1,
        metrics: Optional[List[Metric]] = None,
        dt: float = 0.5,
        maxiter: int = 10000,
        grad_norm_tol: float = 2e-5,
        energy_threshold: float = 100.0,
        hessian_update_interval: int = 10,
    ) -> None:
        super().__init__(backend, log_interval, maxiter, dt, grad_norm_tol)
        if metrics is not None:
            self.metrics: List[Metric] = metrics
        else:
            self.metrics: List[Metric] = []
        self.symmetry_handler: SymmetryHandler = symmetry_handler
        self.subspace_update_interval: int = subspace_update_interval
        self.check_stop_interval = check_stop_interval
        self.subspace_handler: SubspaceHandler = subspace_handler
        self.dt: float = dt
        self.maxiter: int = maxiter
        self.energy_threshold: float = energy_threshold
        self.hess_evals: Optional[np.ndarray] = None
        self.index: Optional[int] = None
        self.subspace: Optional[np.ndarray] = None

        # preconditioner settings
        if hessian_update_interval >= 1:
            self.use_preconditioner = True
            self.hessian_update_interval = hessian_update_interval
        else:
            self.hessian_update_interval = None
            self.use_preconditioner = False

        # dynamic trust radius and bounce detection state settings
        self.initial_max_allowed_step = 0.05
        self.current_max_step = self.initial_max_allowed_step
        self.prev_f = None
        self.priority_ratio = 0.6
        self.consecutive_safe_steps = 0
        self.consecutive_safe_steps_after_increase = 0
        self.prev_xdot_descent = None
        self.prev_f_descent = None
        
        # check for fragmentation every 100 iterations
        self.check_fragmentation_interval = 100
        # keep maximum energy for the current search for the barrierless dissociation test
        self.max_energy = -1000000
        self.cur_energy = None

    def _write_debug_csv(
        self,
        step: int,
        f: np.ndarray,
        f_ascent_coeffs: Optional[np.ndarray],
        f_descent: np.ndarray,
        xdot_raw: np.ndarray,
        xdot_final: np.ndarray,
    ) -> None:
        """Write per-step dynamics diagnostics to the primary debug CSV."""
        file_exists = os.path.isfile(self.debug_log_file)

        grad_norm_total = np.linalg.norm(f)
        grad_norm_ascent = np.linalg.norm(f_ascent_coeffs) if f_ascent_coeffs is not None else 0.0
        grad_norm_descent = np.linalg.norm(f_descent)

        max_raw_step = np.max(np.abs(xdot_raw))
        max_final_step = np.max(np.abs(xdot_final))
        is_clamped = max_raw_step > self.current_max_step

        max_descent_force = np.max(np.abs(f_descent))
        amplification_factor = max_raw_step / (max_descent_force + 1e-12)

        row_data = {
            'Step': step,
            'State': self.state,
            'Energy': self.backend.compute_energy(self.state),
            'Trust_Radius': self.current_max_step,
            'Grad_Norm_Total': f"{grad_norm_total:.2e}",
            'Grad_Norm_Ascent': f"{grad_norm_ascent:.2e}",
            'Grad_Norm_Descent': f"{grad_norm_descent:.2e}",
            'Max_Force_Descent': f"{max_descent_force:.2e}",
            'Max_Raw_Step': f"{max_raw_step:.4f}",
            'Amplification_Factor': f"{amplification_factor:.1f}",
            'Was_Clamped': is_clamped,
            'Safe_Steps': self.consecutive_safe_steps,
        }

        with open(self.debug_log_file, mode='a', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    def _make_step(self, state: np.ndarray, step: int) -> np.ndarray:
        f: np.ndarray = -self.backend.compute_gradient(state)
        k = self.index

        if (self.use_preconditioner and step % self.hessian_update_interval == 0):
            if step == 0:
                self.logger.info("Constructing initial static descent preconditioner...")
            else:
                self.logger.info(f"Step {step}: Periodically rebuilding exact descent preconditioner...")
                
            self.P_descent = self.subspace_handler.construct_static_descent_preconditioner(state, k, subspace=self.subspace)
            
            self.prev_f_descent = None
            self.prev_xdot_descent = None

        if k > 0 and self.subspace is not None:
            V = self.subspace
            f_ascent_coeffs = V.T @ f
            f_descent = f - V @ f_ascent_coeffs
        else:
            V = None
            f_descent = f.copy()
                    
                    
        if k > 0 and V is not None and self.hess_evals is not None:
            if self.use_preconditioner:
                ascent_scales = 1.0 / (np.abs(self.hess_evals) + 1e-6)
            else:
                ascent_scales = np.ones(k)
            xdot_ascent = V @ (ascent_scales * f_ascent_coeffs)
            xdot_descent = self.P_descent @ f_descent
            
            # enforce strict orthogonality to V
            xdot_descent = xdot_descent - V @ (V.T @ xdot_descent)
            xdot = xdot_descent - xdot_ascent
        else:
            xdot_descent = self.P_descent @ f_descent
            xdot = xdot_descent.copy()

        self.prev_f_descent = f_descent.copy()
        self.prev_xdot_descent = xdot_descent.copy()

        # bounce detection
        if self.use_preconditioner and self.prev_f is not None:
            flipped_mask = (f * self.prev_f) < 0
            current_max_force = np.max(np.abs(self.prev_f))
            
            priority_threshold = self.priority_ratio * current_max_force
            priority_mask = np.abs(self.prev_f) > priority_threshold
            
            problematic_flips = flipped_mask & priority_mask
            
            if np.any(problematic_flips):
                self.current_max_step *= 0.5
                self.current_max_step = max(self.current_max_step, 0.0005)
                self.logger.info(f"Overshoot! Flips: {np.sum(problematic_flips)} | Max force: {current_max_force:.2e}. Shrinking max_step to {self.current_max_step:.6f}")
                    
        self.prev_f = f.copy()
        
        xdot_raw = xdot.copy()
        
        # apply dynamic trust radius
        N = len(xdot) // 3
        atomic_displacements = xdot.reshape(N, 3)
        atom_step_norms = np.linalg.norm(atomic_displacements, axis=1)
        max_atom_step = np.max(atom_step_norms)

        if self.use_preconditioner and max_atom_step > self.current_max_step:
            xdot = xdot * (self.current_max_step / max_atom_step)

        if step % 10 == 0:
            # We must pass f_ascent_coeffs safely in case we are in a pure k=0 search
            ascent_data = f_ascent_coeffs if (k > 0 and V is not None) else None
            self._write_debug_csv(step, f, ascent_data, f_descent, xdot_raw, xdot)

        self.actual_step = self.dt * xdot
        return state + self.actual_step
    

    def _update(self, step: int) -> None:
        self.cur_energy = self.backend.compute_energy(self.state)
        self.max_energy = max(self.max_energy, self.cur_energy)

        if step % self.subspace_update_interval == 0:
            new_evals, new_evecs = self.subspace_handler.renew_subspace(
                subspace=self.eigen_subspace,
                state=self.state
            )

            if new_evecs is not None and new_evecs.size > 0:
                self.eigen_subspace = new_evecs

            if self.index > 0 and self.eigen_subspace is not None:
                self.subspace = self.eigen_subspace
            
            self.hess_evals = new_evals


    def _check_stopping_criterion(self, step: int) -> tuple[bool, bool]:
        if self.check_stop_interval is None or step % self.check_stop_interval != 0:
            return (False, False)
        grad: np.ndarray = self.backend.compute_gradient(self.state)
        grad_norm: float = np.linalg.norm(grad)
        if grad_norm < self.grad_norm_tol:
            self.logger.debug(f'Stopping criterion reached: Grad_norm: {grad_norm}')
            return (True, True)
        #energy = self.backend.compute_energy(self.state)
        if self.cur_energy > self.energy_threshold:
            self.logger.error(f'Energy exploded: {self.cur_energy} > {self.energy_threshold}. Aborting run.')
            return (True, False)

        # fragmentation check
        if step % self.check_fragmentation_interval == 0:
            num_fragments = count_fragments(self.backend.get_xyz(self.state), mode='vdw')
            if num_fragments > 1:
                energy_drop = self.max_energy - self.cur_energy
                
                # tolerance to account for orthogonal relaxation noise
                if energy_drop < 1e-4: 
                    self.logger.warning(f"Barrierless dissociation detected at step {step}. Flagging as product.")
                    self.logger.info(f"Final state {self.backend.get_xyz(self.state)}")
                    return (True, True)
                else:
                    self.logger.warning(f"Fragmented, but energy dropped by {energy_drop:.4f}. Discard the result.")
                    self.logger.info(f"Final state {self.backend.get_xyz(self.state)}")
                    return (True, False)

        return (False, False)

    def _build_critical_point(self, state: np.ndarray) -> CriticalPoint:
        return CriticalPoint.from_state(state, backend=self.backend)

    def run(self, run_condition: RunCondition) -> Optional[CriticalPoint]:
        self.index = run_condition.index
        self.state = run_condition.get_perturbed_state()

        self.eigen_subspace = run_condition.subspace

        if self.index > 0:
            self.subspace = self.eigen_subspace
            if self.subspace is not None and self.subspace.shape[1] != self.index:
                self.logger.warning(
                    f"subspace has {self.subspace.shape[1]} columns "
                    f"but index={self.index}; expected them to match."
                )
        else:
            self.subspace = None

        self.current_max_step = self.initial_max_allowed_step
        self.prev_f = None
        self.priority_ratio = 0.8
        
        self.P_descent = np.eye(self.state.shape[0])
        self.prev_xdot_descent = None
        self.prev_f_descent = None
        self.hess_evals = None
        self.actual_step = None
        
        self.initial_energy = self.backend.compute_energy(self.state)
        self.cur_energy = self.initial_energy
        self.max_energy = self.initial_energy
        
        job_label = os.environ.get("HISD_JOB_ID", "serial")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('./debug_info', exist_ok=True)
        self.debug_log_file = f"./debug_info/job_info_{job_label}_{timestamp}.csv"
        
        self._update(0)
        for step in range(self.maxiter):
            self.state = self._make_step(self.state, step+1)
            self._update(step+1)
            self._log(step)
                        
            should_stop, is_success = self._check_stopping_criterion(step)
            if should_stop:
                if is_success:
                    add_edge = bool((np.abs(self.max_energy - self.cur_energy) < 1e-4 ) or
                                    (np.abs(self.initial_energy - self.max_energy) < 1e-4))
                    return (self._build_critical_point(self.state), add_edge)
                else:
                    self.logger.warning(f'Aborted run as energy exceeded tolerance ({self.energy_threshold}) or the molecule fragmented.')
                    return (None, None)
        self.logger.warning(f'Reached maximum iterations ({self.maxiter}) without finding a critical point!')
        return (None, None)
