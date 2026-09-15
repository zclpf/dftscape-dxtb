from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import torch

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

# Constants
BOHR_TO_ANGSTROM = 0.52917721092

@BackendRegistry.register("dxtb")
class DxtbBackend(Backend):
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
        # Check availability and import dxtb
        try:
            import dxtb
            import dxtb.calculators
        except ImportError:
            raise RuntimeError(
                "dxtb is not installed. Please install it to use this backend."
            )

        self.cmp_mthd: str = cmp_mthd
        self.memory: str = memory
        self.num_threads: int = num_threads
        self.options: Dict[str, Any] = options.copy()
        
        # Ensure strict SCF convergence by default (critical for Hessians)
        # We inject this into options if not present
        self.options = {
                "maxiter": 200, 
                "x_atol": 1e-9, 
                "f_atol": 1e-9,
                "verbosity": 0
            }
        
        # PyTorch configuration: Force CPU as requested
        self.device = torch.device("cpu")
        self.dtype = torch.double
        torch.set_num_threads(num_threads)

        # Parse Molecule XYZ to Numbers and Positions
        self.numbers, self.positions = self._parse_xyz(molecule_xyz)
        self.molecule_directives = molecule_directives
        
        # Resolve Method (e.g., "GFN1-xTB" -> dxtb.GFN1_XTB)
        method_map = {
            "GFN1-xTB": dxtb.GFN1_XTB,
            "GFN2-xTB": dxtb.GFN2_XTB,
            "gfn1-xtb": dxtb.GFN1_XTB,
            "gfn2-xtb": dxtb.GFN2_XTB,
        }
        
        if self.cmp_mthd not in method_map:
             self.parametrization = dxtb.GFN1_XTB 
        else:
            self.parametrization = method_map[self.cmp_mthd]

        # Initialize Calculator
        self.calculator = dxtb.Calculator(
            self.numbers, 
            self.parametrization, 
            device=self.device, 
            dtype=self.dtype,
            opts=self.options
        )
        

    def _parse_xyz(self, xyz: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parses XYZ string to atomic numbers (Tensor) and positions (Tensor in Bohr)."""
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
            # XYZ is in Angstrom, convert to Bohr for internal storage/calculation
            coords = [float(x) for x in parts[1:4]]
            coords_list.append(coords)

        numbers = torch.tensor(numbers_list, device=self.device, dtype=torch.long)
        # Convert Angstrom to Bohr
        positions = torch.tensor(coords_list, device=self.device, dtype=self.dtype) / BOHR_TO_ANGSTROM
        return numbers, positions

    def __del__(self) -> None:
        pass

    # ---------------------------- compute ----------------------------

    def compute_energy(self, state: np.ndarray) -> float:
        self._set_state(state)
        with torch.no_grad():
            pos_tensor = self.positions.clone().detach()
            energy = self.calculator.get_energy(pos_tensor)
            return float(energy.item())

    def compute_gradient(self, state: np.ndarray) -> np.ndarray:
        self.calculator.reset()
        self._set_state(state)
        pos_tensor = self.positions.clone().detach().requires_grad_(True)
        
        # Compute energy
        energy = self.calculator.get_energy(pos_tensor)
        
        # Compute gradient using autograd
        grad = torch.autograd.grad(energy, pos_tensor, create_graph=False)[0]
        grad_np = grad.detach().cpu().numpy().ravel()
        
        del energy
        del grad
        pos_tensor.requires_grad_(False)
        
        return grad_np

    def compute_hessian(self, state: np.ndarray) -> np.ndarray:
        self.calculator.reset()
        self._set_state(state)
        
        # IMPORTANT: Detach charges from previous calculations (like optimization)
        # to ensure the Hessian starts with a clean graph history.
        if hasattr(self.calculator, "charges") and self.calculator.charges is not None:
             self.calculator.charges = None
        if hasattr(self.calculator, "potential") and self.calculator.potential is not None:
             self.calculator.potential = None

        # Requires grad is needed for the internal graph building in dxtb.hessian
        pos_tensor = self.positions.clone().detach().requires_grad_(True)
        
        # Use built-in dxtb Hessian
        hess_tensor = self.calculator.hessian(pos_tensor)
        
        # Reshape to (3N, 3N) standard matrix format
        n_atoms = self.numbers.shape[0]
        hess_tensor = hess_tensor.reshape(n_atoms * 3, n_atoms * 3)
        hess = hess_tensor.detach().cpu().numpy()
        
        del hess_tensor
        pos_tensor.requires_grad_(False)
        
        return (hess+hess.T)/2

    # ---------------------------- state ----------------------------

    def _set_state(self, state: np.ndarray) -> None:
        coords = np.asarray(state, dtype=float).reshape(-1, 3)
        if coords.shape[0] != self.numbers.shape[0]:
            raise ValueError(
                f"State has {coords.shape[0]} atoms but molecule has {self.numbers.shape[0]}."
            )
        self.positions = torch.tensor(coords, device=self.device, dtype=self.dtype)

    # ---------------------------- I/O ----------------------------

    def get_xyz(self, state: np.ndarray) -> str:
        # Convert state (Bohr) to Angstrom for XYZ output
        coords = np.asarray(state, dtype=float).reshape(-1, 3) * BOHR_TO_ANGSTROM
        
        natoms = len(self.numbers)
        lines = [f"{natoms}", "Generated by DxtbBackend"]
        
        numbers_np = self.numbers.cpu().numpy()
        for i in range(natoms):
            z = numbers_np[i]
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
        
        return np.array(coords_list) / BOHR_TO_ANGSTROM

    def optimize(self, state: np.ndarray) -> np.ndarray:
        """
        Optimizes geometry using PyTorch L-BFGS (Robust for dxtb).
        """
        self._set_state(state)
        # Clone ensures we don't mess up internal state if opt fails
        pos_opt = self.positions.clone().detach().requires_grad_(True)
        
        # 'strong_wolfe' line search is CRITICAL for molecular potentials
        optimizer = torch.optim.LBFGS(
            [pos_opt], 
            lr=1.0, 
            max_iter=50,       
            tolerance_grad=1e-6, # Strict tolerance 
            tolerance_change=1e-9, 
            history_size=20, 
            line_search_fn="strong_wolfe" 
        )

        def closure():
            optimizer.zero_grad()
            
            # Detach guesses to prevent graph errors during opt
            if hasattr(self.calculator, "charges") and self.calculator.charges is not None:
                self.calculator.charges = self.calculator.charges.detach()
            if hasattr(self.calculator, "potential") and self.calculator.potential is not None:
                self.calculator.potential = self.calculator.potential.detach()
            
            try:
                energy = self.calculator.get_energy(pos_opt)
                energy.backward()
                return energy
            except RuntimeError:
                # Fallback: reset if SCF fails
                self.calculator.reset()
                energy = self.calculator.get_energy(pos_opt)
                energy.backward()
                return energy

        # Run Optimization Loop
        for i in range(200):
            try:
                optimizer.step(closure)
            except RuntimeError:
                self.calculator.reset()
                continue
            
            # Check convergence safely
            if pos_opt.grad is not None:
                grad_norm = pos_opt.grad.norm().item()
                if grad_norm < 1e-5:
                    break
            else:
                # Force re-eval if gradient missing
                with torch.enable_grad():
                    closure()

        # Update internal state and return flattened array
        self.positions = pos_opt.detach()
        return self.positions.cpu().numpy().ravel()
