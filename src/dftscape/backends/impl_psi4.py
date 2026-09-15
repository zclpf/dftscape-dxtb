from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
import tempfile
import os

import numpy as np

from .registry import BackendRegistry
from ..core.interfaces import Backend

if TYPE_CHECKING:
    # For type hints only; not imported at runtime
    import psi4  # noqa: F401


@BackendRegistry.register("psi4")
class Psi4Backend(Backend):

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
        # Check availability and import psi4
        try:
            import psi4  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Psi4 is not installed. Follow instructions in README.md to install it."
            )

        # Store configuration parameters as instance attributes
        self.cmp_mthd: str = cmp_mthd
        self.memory: str = memory
        self.num_threads: int = num_threads
        self.options: Dict[str, Any] = (
            options.copy()
        )  # Make a copy to avoid external modifications

        # Configure Psi4 global settings
        psi4.set_memory(memory)
        psi4.set_num_threads(num_threads)
        for key, value in options.items():
            psi4.set_options({key: value})

        # Handle output file based on use_temp_file option
        if use_temp_file:
            # Use a temporary file to suppress output files in working directory
            temp_output: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".out"
            )
            temp_output.close()  # Close it so psi4 can write to it
            psi4.set_output_file(temp_output.name)
            self.temp_output_file: str = temp_output.name  # Store the temp file path
        else:
            # Redirect output to /dev/null to discard all output
            # Create a symlink to /dev/null to avoid .log suffix issues
            import uuid

            null_link = f"/tmp/psi4_null_{uuid.uuid4().hex}"
            try:
                os.symlink("/dev/null", null_link)
                psi4.set_output_file(null_link)
                self.temp_output_file: str = null_link  # Store for cleanup
            except OSError:
                # Fallback: use a temporary file that gets deleted
                temp_null = tempfile.NamedTemporaryFile(delete=False)
                temp_null.close()
                os.unlink(temp_null.name)
                psi4.set_output_file(temp_null.name)
                self.temp_output_file: Optional[str] = None
        self.wfn: Optional[Any] = None  # Psi4 wavefunction object
        self.molecule: Any = psi4.core.Molecule.from_string(
            molecule_xyz, **molecule_directives
        )
        self.molecule_directives: Dict[str, Any] = molecule_directives

    def __del__(self) -> None:
        try:
            if (
                hasattr(self, "temp_output_file")
                and self.temp_output_file is not None
                and os.path.exists(self.temp_output_file)
            ):
                # Handle both regular files and symlinks
                if os.path.islink(self.temp_output_file):
                    os.unlink(self.temp_output_file)  # Remove symlink
                else:
                    os.unlink(self.temp_output_file)  # Remove regular file
        except (OSError, AttributeError):
            # Ignore cleanup errors during shutdown
            pass

    # ---------------------------- compute ----------------------------

    def compute_energy(self, state: np.ndarray) -> float:
        import psi4  # type: ignore

        self._set_state(state)
        kwargs: Dict[str, Any] = {"return_wfn": True, "molecule": self.molecule}
        if self.wfn is not None:
            kwargs["guess"] = self.wfn
        energy: float
        energy, self.wfn = psi4.energy(self.cmp_mthd, **kwargs)
        return float(energy)

    def compute_gradient(self, state: np.ndarray) -> np.ndarray:
        import psi4  # type: ignore

        self._set_state(state)
        kwargs: Dict[str, Any] = {"return_wfn": True, "molecule": self.molecule}
        if self.wfn is not None:
            kwargs["guess"] = self.wfn
        grad: Any
        grad, self.wfn = psi4.gradient(self.cmp_mthd, **kwargs)
        # Psi4 returns a Matrix-like; convert to flat numpy array
        return np.asarray(grad.to_array(), dtype=float).ravel()

    def compute_hessian(self, state: np.ndarray) -> np.ndarray:
        import psi4  # type: ignore

        self._set_state(state)
        kwargs: Dict[str, Any] = {"return_wfn": True, "molecule": self.molecule}
        hess: Any
        hess, self.wfn = psi4.hessian(self.cmp_mthd, **kwargs)
        return np.asarray(hess.to_array(), dtype=float)

    # ---------------------------- state ----------------------------

    def _set_state(self, state: np.ndarray) -> None:
        import psi4  # type: ignore

        coords: np.ndarray = np.asarray(state, dtype=float).reshape(-1, 3)
        if self.molecule.natom() == 0:
            raise RuntimeError(
                "Psi4 Molecule has 0 atoms. Initialise or load a molecule before set_state()."
            )
        if coords.shape[0] != self.molecule.natom():
            raise ValueError(
                f"State has {coords.shape[0]} atoms but molecule has {self.molecule.natom()}."
            )
        geom_mat: Any = psi4.core.Matrix.from_array(coords)
        self.molecule.set_geometry(geom_mat)
        # Keep internal caches consistent (bond lengths, inertia, etc.)
        self.molecule.update_geometry()

    # ---------------------------- I/O ----------------------------

    def get_xyz(self, state: np.ndarray) -> str:
        self._set_state(state)
        return self.molecule.save_string_xyz_file()

    def get_state_from_xyz(self, xyz: str) -> np.ndarray:
        from psi4 import constants as pc

        lines = xyz.strip().splitlines()
        if len(lines) < 3:
            raise ValueError("XYZ string is too short to contain valid geometry.")
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
        return np.array(coords_list) / pc.bohr2angstroms
    
    def optimize(self, state: np.ndarray) -> np.ndarray:
        import psi4  # type: ignore

        self._set_state(state)
        kwargs: Dict[str, Any] = {"return_wfn": True, "molecule": self.molecule}
        if self.wfn is not None:
            kwargs["guess"] = self.wfn
        psi4.set_options({
            "geom_maxiter": 50,                
            "g_convergence": "gau_verytight",      
        })
        
        energy, self.wfn = psi4.optimize(self.cmp_mthd, **kwargs)

        return self.wfn.molecule().geometry().to_array()
