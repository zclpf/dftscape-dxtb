from __future__ import annotations
import numpy as np
import logging
from typing import Tuple, Optional
from .interfaces import Backend
from .symmetry_handlers import SymmetryHandler
from .eigensolvers import Eigensolver, EigensolverRegistry


class SubspaceHandler:

    def __init__(self, backend: Backend, symmetry_handler: SymmetryHandler, eigensolver: Eigensolver):
        self.backend = backend
        self.symmetry_handler = symmetry_handler
        self.eigensolver = eigensolver
        self.logger = logging.getLogger(__name__)

        eigensolver_kwargs = {'backend': backend, 'k': 10000000, 'tol': 1e-3, 'maxiter': 50}
        self.preconditioner_eigensolver = EigensolverRegistry.create("eigh", **eigensolver_kwargs)
        self.preconditioner_eps = 1e-6

    
    def renew_subspace(self, 
                       subspace: np.ndarray,
                       state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Always run eigensolver, even if k=0, to cache the preconditioner Hessian
        k = subspace.shape[1] if subspace is not None else 0
        self.eigensolver.k = k

        constraint_subspace = self.symmetry_handler.get_constraint_subspace(state)
        self.logger.debug(f'Constraint subspace: {(constraint_subspace.shape if constraint_subspace is not None else None)}')
        Y = constraint_subspace if constraint_subspace is not None and constraint_subspace.shape[1] > 0 else None

        eigenvalues, eigenvectors = self.eigensolver.solve(state, X0=subspace, Y=Y)

        if eigenvalues is None or eigenvectors is None:
            self.logger.debug('Eigensolver returned None, using gradient flow (subspace 0)')
            return (np.array([]), np.array([]).reshape(state.shape[0], 0))

        self.logger.debug(f'Final eigenvectors shape: {eigenvectors.shape}')
        self.logger.debug(f'Final eigenvalues: {eigenvalues}')
        return (eigenvalues, eigenvectors) 
     

    def construct_static_descent_preconditioner(self, state: np.ndarray, k: int, subspace: Optional[np.ndarray] = None) -> np.ndarray:
        n = state.shape[0]
        self.preconditioner_eigensolver.k = n

        constraint_subspace = self.symmetry_handler.get_constraint_subspace(state)
        self.logger.debug(f'[Preconditioner] Constraint subspace: {(constraint_subspace.shape if constraint_subspace is not None else None)}')
        Y = constraint_subspace if constraint_subspace is not None and constraint_subspace.shape[1] > 0 else None
        
        # Calculate full exact Hessian for the static preconditioner
        eigenvalues, eigenvectors = self.preconditioner_eigensolver.solve(state, X0=subspace, Y=Y)
        if eigenvalues is None or eigenvectors is None:
            self.logger.warning('Preconditioner eigensolver failed, returning identity matrix.')
            return np.identity(n)
        
        # filter out BOTH the structural zero-eigenvalues AND the 1e6 penalty-shifted constraints
        zero_threshold = 1e-8        
        valid_mask = (np.abs(eigenvalues) > zero_threshold) & (np.abs(eigenvalues) < 1e5)
        
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_eigenvectors = eigenvectors[:, valid_mask] 
        
        # construct the FULL preconditioner matrix P for all valid modes
        scaling = 1.0 / (np.abs(valid_eigenvalues) + self.preconditioner_eps)
        P_full = (valid_eigenvectors * scaling) @ valid_eigenvectors.T
        
        if k > 0 and subspace is not None:
            # -------------------------------------------------------------
            # THE FIX: Mathematical Subspace Projection
            # -------------------------------------------------------------
            # Subspace is shape (n, k) and its columns are orthonormal.
            # The projector onto the subspace is P_sub = subspace @ subspace.T
            # The projector onto the orthogonal complement is P_ortho = I - P_sub
            
            I = np.eye(n)
            P_sub = subspace @ subspace.T
            P_ortho = I - P_sub
            
            # Project the full preconditioner so it has exactly 0 effect 
            # along any vector in the k-dimensional subspace!
            P_descent = P_ortho @ P_full @ P_ortho
            
            self.logger.debug(f'[Preconditioner] Applied projection to remove {k} ascent modes.')
        else:
            P_descent = P_full
        
        return P_descent