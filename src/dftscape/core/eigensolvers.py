from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg, eigsh
import logging

from .interfaces import Backend
from ..common import BaseRegistry


class EigensolverRegistry(BaseRegistry):
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> "Eigensolver":
        return super().create(name, *args, **kwargs)


@dataclass
class Eigensolver(ABC):

    backend: Backend
    k: int
    tol: float = 1e-3
    maxiter: int = 10
    largest: bool = False
    logger: Optional[logging.Logger] = field(default=None, init=False)

    _V_prev: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    # cache of the full spectrum for the preconditioner
    last_full_eigenvalues: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    last_full_eigenvectors: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def solve(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    # TODO: Change to use forward difference (with caching) so that we have 1/2 the computational cost.
    def _centered_difference_hvp(
        self, x: np.ndarray, v: np.ndarray, eps: float
    ) -> np.ndarray:
        grad_plus = self.backend.compute_gradient(x + eps * v)
        grad_minus = self.backend.compute_gradient(x - eps * v)
        return (grad_plus - grad_minus) / (2.0 * eps)

    def _qr_full_rank(self, X: np.ndarray) -> np.ndarray:
        # Special handling for k=0
        if self.k == 0:
            return np.array([]).reshape(X.shape[0], 0)

        Q, R = np.linalg.qr(X)
        r = np.linalg.matrix_rank(R)
        if r < self.k:
            # top-up with random directions orthogonal to Q
            rng = np.random.default_rng(getattr(self, "random_state", 42))
            n = X.shape[0]
            extra = rng.standard_normal((n, self.k - r))
            # orthogonalize against Q
            extra = extra - Q @ (Q.T @ extra)
            Q2, _ = np.linalg.qr(extra)
            Q = np.concatenate([Q[:, :r], Q2[:, : (self.k - r)]], axis=1)
        return Q[:, : self.k]

    def _prepare_solve_inputs(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        x = np.asarray(state, dtype=float)
        n = x.size

        # Allow inputs to be processed even if k=0 so the full Hessian is computed
        if self.k == 0:
            X = np.array([]).reshape(n, 0)
        else:
            X = None
            if X0 is not None:
                X = np.asarray(X0, dtype=float)
                if X.shape != (n, self.k):
                    raise ValueError(f"X0 must have shape {(n, self.k)}, got {X.shape}.")
                X = self._qr_full_rank(X)
            elif (self._V_prev is not None) and (self._V_prev.shape == (n, self.k)):
                X = self._qr_full_rank(self._V_prev)

        # Process Y
        if Y is not None:
            Y = np.asarray(Y, dtype=float)
            if Y.shape[0] != n:
                raise ValueError(f"Y must have {n} rows, got {Y.shape[0]}.")

        return x, X, Y

    def _initialize_subspace(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(getattr(self, "random_state", 42))
        return self._qr_full_rank(rng.standard_normal((n, self.k)))

    def _orthogonalize_against_constraints(
        self, vectors: np.ndarray, constraint_matrix: Optional[np.ndarray]
    ) -> np.ndarray:
        if constraint_matrix is None or constraint_matrix.shape[1] == 0:
            return vectors

        Q, _ = np.linalg.qr(constraint_matrix)
        P = np.eye(vectors.shape[0]) - Q @ Q.T  # Projection onto orthogonal complement
        return P @ vectors

    def _finalize_solve(
        self, vals: np.ndarray, vecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._V_prev = np.array(vecs, copy=True)
        return vals, vecs

    def _compute_hvp_batch(
        self, x: np.ndarray, V: np.ndarray, eps: Optional[float] = None
    ) -> np.ndarray:
        if eps is None:
            eps = getattr(self, "fd_eps", 1e-4)
        GV = np.zeros(V.shape)
        for i in range(V.shape[1]):
            GV[:, i] = self._centered_difference_hvp(x, V[:, i], eps)
        return GV


@dataclass
@EigensolverRegistry.register("lobpcg")
class ScipyLOBPCGEigensolver(Eigensolver):

    # Additional parameters specific to finite-difference LOBPCG
    fd_eps: float = 1e-4
    random_state: Optional[int] = 42

    # ------------------------------ public ------------------------------
    def solve(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, X, Y = self._prepare_solve_inputs(state, X0, Y)
        if x is None:
            return None, None

        n = x.size
        A = self._make_operator(x, n)

        # If X is None, initialize random
        if X is None:
            rng = np.random.default_rng(self.random_state)
            X = self._qr_full_rank(rng.standard_normal((n, self.k)))

        # Try LOBPCG with constraints, fall back to manual constraint handling if needed
        try:
            vals, vecs = lobpcg(
                A, X, Y=Y, largest=self.largest, tol=self.tol, maxiter=self.maxiter
            )

        except NotImplementedError as e:
            if "dense eigensolver does not support constraints" in str(e):
                self.logger.warning(
                    "LOBPCG dense fallback doesn't support constraints, using manual orthogonalization"
                )

                # Solve without constraints
                vals, vecs = lobpcg(
                    A,
                    X,
                    Y=None,
                    largest=self.largest,
                    tol=self.tol,
                    maxiter=self.maxiter,
                )

                # Manually orthogonalize against constraint subspace
                if Y is not None and Y.shape[1] > 0:
                    vecs = self._orthogonalize_against_constraints(vecs, Y)
            else:
                raise

        return self._finalize_solve(vals, vecs)

    # ----------------------------- private -----------------------------
    def _make_operator(self, x: np.ndarray, n: int) -> LinearOperator:

        def hvp_block(V: np.ndarray, eps: float = self.fd_eps) -> np.ndarray:
            V = np.asarray(V, dtype=float)
            if V.ndim == 1:
                V = V.reshape(-1, 1)

            # Normalize columns so |ε v̂| is controlled (stabilizes FD)
            norms = np.linalg.norm(V, axis=0, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            Vn = V / norms

            Gp = np.column_stack(
                [
                    self.backend.compute_gradient(x + eps * Vn[:, i])
                    for i in range(V.shape[1])
                ]
            )
            Gm = np.column_stack(
                [
                    self.backend.compute_gradient(x - eps * Vn[:, i])
                    for i in range(V.shape[1])
                ]
            )
            HVi = (Gp - Gm) / (2.0 * eps)
            return HVi * norms  # undo normalization to keep linearity

        def matvec(v: np.ndarray) -> np.ndarray:
            return hvp_block(v).ravel()

        def matmat(V: np.ndarray) -> np.ndarray:
            return hvp_block(V)

        return LinearOperator(shape=(n, n), matvec=matvec, matmat=matmat, dtype=float)


@dataclass
class EigshEigensolver(Eigensolver):

    def solve(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, _, Y = self._prepare_solve_inputs(state, None, Y)
        if x is None:
            return None, None

        n = x.size

        # Compute exact Hessian matrix
        H = self.backend.compute_hessian(x)

        # eigsh parameters
        k_eigsh = min(self.k, n)  # For small dense matrices, we can get all eigenvalues
        which = "LA" if self.largest else "SA"  # Largest or Smallest Algebraic

        # Prepare constraint matrix for eigsh (different format than lobpcg)
        sigma = None  # shift-invert mode, can help with convergence
        if Y is not None and Y.shape[1] > 0:
            # Project Hessian onto orthogonal complement of Y
            # Note: Unlike iterative solvers, dense solvers handle constraints by projecting the Hessian matrix
            # rather than orthogonalizing the resulting eigenvectors. This ensures exact orthogonality.
            Q, _ = np.linalg.qr(Y)
            P = np.eye(n) - Q @ Q.T  # projection matrix
            H = P @ H @ P

        # Solve using eigsh
        try:
            vals, vecs = eigsh(
                H,
                k=k_eigsh,
                which=which,
                tol=self.tol,
                maxiter=self.maxiter,
                sigma=sigma,
                return_eigenvectors=True,
            )
        except np.linalg.LinAlgError:
            # Fallback to dense eigensolver if eigsh fails
            self.logger.warning(
                "eigsh failed, falling back to dense eigendecomposition"
            )
            vals_all, vecs_all = np.linalg.eigh(H)
            if self.largest:
                idx = np.argsort(vals_all)[-k_eigsh:]
            else:
                idx = np.argsort(vals_all)[:k_eigsh]
            vals, vecs = vals_all[idx], vecs_all[:, idx]

        # Ensure we return exactly k eigenvalues/vectors
        if self.k == 0:
            # For k=0, return empty arrays
            vals = np.array([])
            vecs = np.array([]).reshape(n, 0)
        elif vals.shape[0] < self.k:
            # Pad with zeros if we couldn't get enough eigenvalues
            n_missing = self.k - vals.shape[0]
            vals = np.concatenate([vals, np.zeros(n_missing)])
            vecs = np.concatenate([vecs, np.zeros((n, n_missing))], axis=1)

        # Sort eigenvalues (eigsh may not return them in perfect order)
        if self.k > 0:
            if self.largest:
                sort_idx = np.argsort(vals)[::-1][: self.k]
            else:
                sort_idx = np.argsort(vals)[: self.k]

            vals = vals[sort_idx]
            vecs = vecs[:, sort_idx]

        return self._finalize_solve(vals, vecs)


@dataclass
@EigensolverRegistry.register("eigh")
class EighEigensolver(Eigensolver):

    def solve(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, _, Y = self._prepare_solve_inputs(state, None, Y)
        if x is None:
            return None, None

        n = x.size

        # Compute exact Hessian matrix
        H = self.backend.compute_hessian(x)

        # Prepare constraint matrix
        if Y is not None and Y.shape[1] > 0:
            # Project Hessian onto orthogonal complement of Y
            # Note: Unlike iterative solvers, dense solvers handle constraints by projecting the Hessian matrix
            # rather than orthogonalizing the resulting eigenvectors. This ensures exact orthogonality.
            Q, _ = np.linalg.qr(Y)
            P = np.eye(n) - Q @ Q.T  # projection matrix
            H = P @ H @ P
            
            # PUT THIS BACK: Hide the trivial modes at 1,000,000.0
            shift = -1e6 if self.largest else 1e6
            H = H + shift * (Q @ Q.T)

        # Solve using numpy.linalg.eigh (exact for small dense matrices)
        try:
            vals_all, vecs_all = np.linalg.eigh(H)
        except np.linalg.LinAlgError as e:
            self.logger.error(f"eigh failed: {e}")
            raise

        # Sort eigenvalues and eigenvectors
        if self.largest:
            # Sort in descending order (largest first)
            sort_idx = np.argsort(vals_all)[::-1]
        else:
            # Sort in ascending order (smallest first)
            sort_idx = np.argsort(vals_all)

        vals_all = vals_all[sort_idx]
        vecs_all = vecs_all[:, sort_idx]
        
        # Cache the complete spectrum for use by the preconditioner
        self.last_full_eigenvalues = vals_all
        self.last_full_eigenvectors = vecs_all

        # Return empty/sliced arrays depending on k
        if self.k == 0:
            vals = np.array([])
            vecs = np.array([]).reshape(n, 0)
        elif self.k >= n:
            vals = vals_all
            vecs = vecs_all
        else:
            vals = vals_all[: self.k]
            vecs = vecs_all[:, : self.k]

        return self._finalize_solve(vals, vecs)


@dataclass
@EigensolverRegistry.register("lobpcg_manual")
class LOBPCGManualEigensolver(Eigensolver):

    # Additional parameters specific to finite-difference LOBPCG
    fd_eps: float = 1e-4
    random_state: Optional[int] = 42

    # Internal state for backup vector
    _backup_vector: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _iteration_count: int = field(default=0, init=False, repr=False)
    _reset_interval: int = 10  # Reset backup vector every N iterations

    def solve(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, X, Y = self._prepare_solve_inputs(state, X0, Y)
        if x is None:
            return None, None

        n = x.size

        # Initialize subspace
        if X is None:
            rng = np.random.default_rng(self.random_state)
            X = self._qr_full_rank(rng.standard_normal((n, self.k)))
        subspace = X

        # Initialize backup vector if not set
        if self._backup_vector is None or self._backup_vector.shape[0] != n:
            rng = np.random.default_rng(self.random_state)
            self._backup_vector = rng.standard_normal(n)

        # Orthogonalize subspace against constraints
        if Y is not None:
            subspace = self._orthogonalize_against_constraints(subspace, Y)

        # Run manual LOBPCG iterations
        eigenvalues, eigenvectors = self._manual_lobpcg(x, subspace, Y)

        # Update backup vector with next highest eigenvector (k+1)
        if eigenvectors.shape[1] > self.k:
            self._backup_vector = eigenvectors[:, self.k].copy()
        else:
            # If we don't have k+1 eigenvector, compute it
            self._backup_vector = self._compute_next_eigenvector(x, eigenvectors, Y)

        # Increment iteration count and potentially reset backup vector
        self._iteration_count += 1
        if self._iteration_count % self._reset_interval == 0:
            rng = np.random.default_rng(self.random_state)
            self._backup_vector = rng.standard_normal(n)

        return self._finalize_solve(eigenvalues, eigenvectors)

    def _manual_lobpcg(
        self, x: np.ndarray, subspace: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, k = subspace.shape
        V = subspace.copy()

        # Compute Hessian-vector products for current subspace
        GV = self._compute_hessian_vector_products(V, x)

        for iteration in range(self.maxiter):
            # Compute preconditioned residual
            W = self._compute_preconditioned_residual(V, GV)

            # Orthogonalize residual subspace
            W_orth = self._orthogonalize_residual_subspace(W, Y, V)

            # Check convergence
            residual_norm = np.linalg.norm(W_orth)
            if residual_norm < self.tol:
                break

            # Solve projected eigenvalue problem
            VW = np.concatenate([V, W_orth], axis=1)
            GW = self._compute_hessian_vector_products(W_orth, x)

            rayleigh_matrix = VW.T @ np.concatenate([GV, GW], axis=1)
            rayleigh_matrix = (rayleigh_matrix + rayleigh_matrix.T) / 2

            try:
                eigvals, eigvecs = np.linalg.eigh(rayleigh_matrix)
            except np.linalg.LinAlgError:
                # Fallback: use eigvals instead
                eigvals, eigvecs = np.linalg.eig(rayleigh_matrix)
                # Sort by real part
                idx = np.argsort(eigvals.real)
                eigvals = eigvals[idx].real
                eigvecs = eigvecs[:, idx].real

            # Select smallest or largest eigenvalues
            if self.largest:
                selected_idx = np.argsort(eigvals)[-k:]
            else:
                selected_idx = np.argsort(eigvals)[:k]

            eigvecs_selected = eigvecs[:, selected_idx]

            # Update subspace
            V_new = VW @ eigvecs_selected

            # Re-orthogonalize
            V_new = self._qr_full_rank(V_new)

            # Check for convergence
            subspace_diff = np.linalg.norm(V_new - V)
            if subspace_diff < self.tol:
                V = V_new
                break

            V = V_new
            GV = self._compute_hessian_vector_products(V, x)

        # Final eigenvalue computation
        GV_final = self._compute_hessian_vector_products(V, x)
        rayleigh_final = V.T @ GV_final
        rayleigh_final = (rayleigh_final + rayleigh_final.T) / 2

        try:
            final_eigvals, final_eigvecs = np.linalg.eigh(rayleigh_final)
        except np.linalg.LinAlgError:
            final_eigvals, final_eigvecs = np.linalg.eig(rayleigh_final)
            idx = np.argsort(final_eigvals.real)
            final_eigvals = final_eigvals[idx].real
            final_eigvecs = final_eigvecs[:, idx].real

        if self.largest:
            idx = np.argsort(final_eigvals)[-k:]
        else:
            idx = np.argsort(final_eigvals)[:k]

        return final_eigvals[idx], V @ final_eigvecs[:, idx]

    def _orthogonalize_against_constraints(
        self, subspace: np.ndarray, constraint_matrix: Optional[np.ndarray]
    ) -> np.ndarray:
        if constraint_matrix is None or constraint_matrix.shape[1] == 0:
            return subspace

        result = subspace.copy()
        for i in range(result.shape[1]):
            orthogonal_basis = np.concatenate(
                [constraint_matrix, result[:, :i]], axis=1
            )
            result[:, i] = self._gram_schmidt_step(result[:, i], orthogonal_basis)
        return result

    def _gram_schmidt_step(self, vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
        if basis.size == 0:
            norm = np.linalg.norm(vector)
            return vector / norm if norm > 0 else vector

        projection = basis @ (basis.T @ vector)
        orthogonal_vector = vector - projection
        norm = np.linalg.norm(orthogonal_vector)

        if norm < 1e-12:
            # Use backup vector if orthogonalization fails
            orthogonal_vector = self._backup_vector.copy()
            orthogonal_vector = orthogonal_vector - basis @ (
                basis.T @ orthogonal_vector
            )
            norm = np.linalg.norm(orthogonal_vector)
            if norm < 1e-12:
                # Last resort: add small random perturbation
                orthogonal_vector = (
                    orthogonal_vector + np.random.random(vector.shape) * 1e-8
                )
                orthogonal_vector = orthogonal_vector - basis @ (
                    basis.T @ orthogonal_vector
                )
                norm = np.linalg.norm(orthogonal_vector)

        return orthogonal_vector / norm if norm > 0 else orthogonal_vector

    def _compute_hessian_vector_products(
        self, vectors: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        GV = np.zeros(vectors.shape)

        for i in range(vectors.shape[1]):
            GV[:, i] = self._centered_difference_hvp(state, vectors[:, i], self.fd_eps)

        return GV

    def _compute_preconditioned_residual(
        self, subspace: np.ndarray, GV: np.ndarray
    ) -> np.ndarray:
        VTV_GV = np.sum(subspace * GV, axis=0)
        return GV - subspace * VTV_GV

    def _orthogonalize_residual_subspace(
        self,
        W: np.ndarray,
        constraint_matrix: Optional[np.ndarray],
        subspace: np.ndarray,
    ) -> np.ndarray:
        result = W.copy()

        # Build full constraint matrix
        constraints = []
        if constraint_matrix is not None and constraint_matrix.shape[1] > 0:
            constraints.append(constraint_matrix)
        constraints.append(subspace)
        full_constraint = np.concatenate(constraints, axis=1)

        for i in range(result.shape[1]):
            constraint_with_previous = np.concatenate(
                [full_constraint, result[:, :i]], axis=1
            )
            result[:, i] = self._gram_schmidt_step(
                result[:, i], constraint_with_previous
            )

        return result

    def _compute_next_eigenvector(
        self, x: np.ndarray, eigenvectors: np.ndarray, Y: Optional[np.ndarray]
    ) -> np.ndarray:
        # Use random vector orthogonal to current eigenvectors
        rng = np.random.default_rng(self.random_state)
        candidate = rng.standard_normal(x.shape[0])

        # Orthogonalize against current eigenvectors and constraints
        basis = eigenvectors.copy()
        if Y is not None and Y.shape[1] > 0:
            basis = np.concatenate([Y, basis], axis=1)

        return self._gram_schmidt_step(candidate, basis)


@dataclass
@EigensolverRegistry.register("lobpcg_simplified")
class LOBPCGSimplifiedEigensolver(Eigensolver):
    # Additional parameters specific to finite-difference LOBPCG
    fd_eps: float = 1e-4
    random_state: Optional[int] = 42

    def solve(
        self,
        state: np.ndarray,
        X0: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, X, Y = self._prepare_solve_inputs(state, X0, Y)
        if x is None:
            return None, None

        n = x.size

        # Initialize subspace V
        if X is not None:
            V = X
        elif (self._V_prev is not None) and (self._V_prev.shape == (n, self.k)):
            V = self._V_prev.copy()
        else:
            rng = np.random.default_rng(self.random_state)
            V = rng.standard_normal((n, self.k))

        # Orthogonalize V against constraints Y
        if Y is not None:
            V = self._orthogonalize_against_constraints_simple(V, Y)

        # Compute Hessian-vector products for V
        GV = self._compute_hvp_batch(x, V)

        # Simple preconditioning: W = GV - V * (V.T @ GV)
        W = GV - V * np.einsum("ij,ij->j", V, GV)

        # Orthogonalize W against constraints and V
        if Y is not None:
            W = self._orthogonalize_against_constraints_simple(W, Y)
        W = self._orthogonalize_against_constraints_simple(W, V)

        # Handle near-zero W vectors by replacing with random backup
        for i in range(W.shape[1]):
            if np.linalg.norm(W[:, i]) < 1e-6:
                rng = np.random.default_rng(self.random_state)
                # Create backup vector orthogonal to constraints and current subspace
                backup = rng.standard_normal(n)
                if Y is not None:
                    backup = backup - Y @ (Y.T @ backup)
                backup = backup - V @ (V.T @ backup)
                backup = backup - W[:, :i] @ (W[:, :i].T @ backup) if i > 0 else backup
                backup = backup / np.linalg.norm(backup)
                W[:, i] = backup

        # Rayleigh-Ritz procedure
        VW = np.concatenate([V, W], axis=1)
        GW = self._compute_hvp_batch(x, W)

        # Build reduced matrix: RR = VW.T @ [GV, GW]
        RR = VW.T @ np.concatenate([GV, GW], axis=1)
        RR = (RR + RR.T) / 2  # Symmetrize

        # Solve reduced eigenvalue problem
        Reigenvalues, Reigenvectors = np.linalg.eigh(RR)

        # Extract k eigenvectors and eigenvalues
        eigenvalues = Reigenvalues[: self.k]
        eigenvectors = VW @ Reigenvectors[:, : self.k]

        return self._finalize_solve(eigenvalues, eigenvectors)

    def _orthogonalize_against_constraints_simple(
        self, vectors: np.ndarray, constraints: np.ndarray
    ) -> np.ndarray:
        result = vectors.copy()
        for i in range(result.shape[1]):
            # Orthogonalize against all previous vectors and constraints
            basis = (
                np.concatenate([constraints, result[:, :i]], axis=1)
                if i > 0
                else constraints
            )
            if basis.shape[1] > 0:
                result[:, i] = result[:, i] - basis @ (basis.T @ result[:, i])
            # Normalize
            norm = np.linalg.norm(result[:, i])
            if norm > 1e-12:
                result[:, i] = result[:, i] / norm
            else:
                # Replace with random vector if normalization fails
                rng = np.random.default_rng(self.random_state)
                result[:, i] = rng.standard_normal(result.shape[0])
                if basis.shape[1] > 0:
                    result[:, i] = result[:, i] - basis @ (basis.T @ result[:, i])
                result[:, i] = result[:, i] / np.linalg.norm(result[:, i])

        return result