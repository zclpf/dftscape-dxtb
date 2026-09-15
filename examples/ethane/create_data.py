import dxtb
import dxtb.calculators
import torch
import json
import numpy as np
import sys
from pathlib import Path
from dftscape.core.symmetry_handlers import SymmetryRegistry
from dftscape.common.distance import count_fragments

# --- Constants ---
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
ATOMIC_NUMBERS = {v: k for k, v in PERIODIC_TABLE.items()}
BOHR_TO_ANGSTROM = 0.52917721092

def compute_trivial_subspace(state: np.ndarray) -> np.ndarray:
    num_atoms = state.size // 3
    atomic_coordinates = state.reshape(-1, 3)
    trivial_ev = np.zeros((6, num_atoms * 3))

    for i in range(3):
        trivial_ev[i, i::3] = 1.0

    # Rotations (Eq. 5): rᵢ = eᵢ × rⱼ for each atom j
    x, y, z = atomic_coordinates[:, 0], atomic_coordinates[:, 1], atomic_coordinates[:, 2]
    trivial_ev[3] = np.stack([np.zeros_like(x), -z,              y            ], axis=1).ravel()
    trivial_ev[4] = np.stack([z,                np.zeros_like(y), -x           ], axis=1).ravel()
    trivial_ev[5] = np.stack([-y,               x,               np.zeros_like(z)], axis=1).ravel()

    # SVD: singular values are ordered descending, so degenerate modes are always last
    U, S, _ = np.linalg.svd(trivial_ev.T, full_matrices=False)
    rank = np.sum(S > 1e-8)   # 5 for linear, 6 for non-linear

    if rank < 6:
        pass

    return U[:, :rank]

def main(xyz_path, optim=True):
    device = torch.device("cpu")
    dtype = torch.double
    xyz_path = Path(xyz_path)
    if not xyz_path.exists():
        raise FileNotFoundError(f"File not found: {xyz_path}")

    # --- 1. Load Molecule ---
    with open(xyz_path, "r") as f:
        xyz_content = f.read().strip()
    
    lines = xyz_content.splitlines()
    if len(lines) < 3: raise ValueError("XYZ file too short")

    numbers, coords = [], []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4: continue
        sym = parts[0]
        if sym not in PERIODIC_TABLE: 
             import re
             m = re.match(r"([A-Za-z]+)", sym)
             sym = m.group(1) if m else sym
        numbers.append(PERIODIC_TABLE.get(sym, 6))
        coords.append([float(x) for x in parts[1:4]])

    numbers = torch.tensor(numbers, device=device, dtype=torch.long)
    positions = torch.tensor(coords, device=device, dtype=dtype) / BOHR_TO_ANGSTROM

    # --- 2. Initialize Calculator (Strict Options) ---
    opts = {
        "verbosity": 0,
            "maxiter": 200,
            "f_atol": 1e-9,
            "x_atol": 1e-9,
            "scf_mode": "full"
    }
    
    calculator = dxtb.Calculator(
        numbers, dxtb.GFN1_XTB, device=device, dtype=dtype, opts=opts
    )
    print("Optim: ", optim)
    if optim == "True":
        # --- 3. Strict Optimization ---
        print("🚀 Starting Geometry Optimization...")
        # Add tiny noise to break symmetry
        pos_opt = (positions + torch.randn_like(positions) * 0.01).clone().detach().requires_grad_(True)
    
        optimizer = torch.optim.LBFGS(
            [pos_opt], lr=1.0, max_iter=50, 
            tolerance_grad=1e-6, tolerance_change=1e-9, 
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            
            # Detach guesses to prevent graph errors during opt
            if hasattr(calculator, "charges") and calculator.charges is not None:
                calculator.charges = calculator.charges.detach()
            if hasattr(calculator, "potential") and calculator.potential is not None:
                calculator.potential = calculator.potential.detach()
                
            try:
                energy = calculator.get_energy(pos_opt)
                energy.backward()
                return energy
            except RuntimeError:
                # Fallback: Reset and try again (ensure backward is called!)
                calculator.reset()
                energy = calculator.get_energy(pos_opt)
                energy.backward() # <--- FIXED: Populate .grad attribute
                return energy

        for i in range(200):
            try:
                loss = optimizer.step(closure)
            except RuntimeError as e:
                print(f"Warning: Step failed ({e}). Retrying...")
                calculator.reset()
                continue
                
            # Safety check before accessing .norm()
            if pos_opt.grad is None:
                print("Warning: Gradient is None. Forcing re-evaluation.")
                with torch.enable_grad():
                    closure()

            grad_norm = pos_opt.grad.norm().item()
            
            if i % 10 == 0:
                print(f"   Step {i}: E={loss.item():.6f} |Grad|={grad_norm:.7f}")
            
            if grad_norm < 1e-6:
                print(f"✅ Converged at step {i}.")
                break
                
        # Update positions
        positions = pos_opt.detach()
    
    state = positions.detach().cpu().numpy().reshape(-1) 
    
    # --- 4. Built-in Hessian Calculation ---
    print("\n🧮 Computing Hessian using dxtb.Calculator.hessian()...")
    
    hess_calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, device=device, dtype=dtype, opts=opts)
    pos_hess = positions.clone().detach().requires_grad_(True)
    
    # Calculate Hessian
    hessian_tensor = hess_calc.hessian(pos_hess)
    
    n_atoms = len(numbers)
    hess_np = hessian_tensor.reshape(n_atoms*3, n_atoms*3).detach().cpu().numpy()
    
    # Normal single molecule (or more than two covalent fragments)
    sym_handler = SymmetryRegistry.create("molecular")
    comment = "Molecular symmetry (translation + rotation only)"

    constraint_basis = sym_handler.get_constraint_subspace(state)

    # Build the projector onto the internal space
    I = np.eye(state.size)
    if constraint_basis.size > 0:          # protects against degenerate edge cases
        P_constraint = constraint_basis @ constraint_basis.T
        P_int = I - P_constraint
    else:
        P_int = I.copy()

    # Project the Hessian → forces the constrained eigenvalues to *exactly* 0.0
    hess_np = P_int @ hess_np @ P_int
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(hess_np)
    print("\nLowest 16 Eigenvalues (Target: ~0.0 for first 6):")
    print(eigvals[:16])

    # --- 5. Save Output ---
    grad_tensor = hess_calc.get_forces(pos_hess)
    grad_np = -grad_tensor.detach().cpu().numpy().ravel()
    
    print(f"Gradient norm: {np.linalg.norm(grad_np)}")
    
    energy = hess_calc.get_energy(pos_hess)
    energy = energy.detach().cpu().numpy()

    geom_ang = positions.cpu().numpy().reshape(-1, 3) * BOHR_TO_ANGSTROM
    xyz_str = f"{n_atoms}\nOptimized dxtb Geometry\n" + "\n".join(
        [f"{ATOMIC_NUMBERS.get(int(n), 'X')} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}" 
         for n, c in zip(numbers, geom_ang)]
    )

    output_path = xyz_path.with_suffix('.json')
    with open(output_path, 'w') as out:
        json.dump({
            "index": 0,
            #"energy": float(loss.item()),
            "energy": float(energy),
            "xyz": xyz_str,
            "state": state.tolist(),
            "state_dtype": "float64",
            "gradient": grad_np.tolist(),
            "gradient_dtype": "float64",
            "hessian": hess_np.tolist(),
            "hessian_dtype": "float64",
            "eigenvalues": eigvals.tolist(),
            "eigenvectors": eigvecs.tolist(),
            "eigen_dtype": "float64" 
        }, out, indent=2)

    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3: main(sys.argv[1], sys.argv[2])
    else: print("Usage: python strict_hessian_dxtb.py <file.xyz> do_optimization")
