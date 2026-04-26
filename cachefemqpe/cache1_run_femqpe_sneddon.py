"""
Convergence analysis for Sneddon problem by FEM-QPE with fracture rotations 
using multiple fracture angles within the range [0°, 180°). 
"""
from __future__ import annotations

import numpy as np
import porepy as pp
from fems import FemSneddonModel2d
from fems import FEMComputeSifErrorConvergence

import math

import pickle
import os

save_model = "data/rotation/femqpe/cache_models/so2"
os.makedirs(save_model, exist_ok=True)

a = 0.3
poi = 0.25
shear_modulus = 1.0
lam = 2 * shear_modulus * poi / (1 - 2 * poi)

params = {
    "prepare_simulation": True,
    "material_constants": {"solid": pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)},
    "a": a,
    "domain_size": 1.0,
    "p0": 1e-4,
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.0,
}

use_qpe = True
fem_model = FemSneddonModel2d(params, use_qpe)
fem_model.model_setup_and_sol()

print("=== Initial model run completed. ===")

# Save cache models for different mesh sizes and seeds.
# cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
# cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00725]
# cell_sizes = [0.00625]
theta_rads = [
    0, 15, 30, 45, 60, 75, 
    90, 105, 120, 135, 150, 165,
]
cell_sizes = [0.055]

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===")
    params["meshing_arguments"] = {"cell_size": h}

    use_qpe = True
    fem_model = FemSneddonModel2d(params, use_qpe)
    fem_model.model_setup_and_sol()

    model_sol = FEMComputeSifErrorConvergence(model=fem_model)

    for si, theta_rad in enumerate(theta_rads):
        print(f"\nRunning theta_rad = {theta_rad}")
        params["theta_rad"] = math.radians(90 - theta_rad)

        # Run model. 
        fem_model = FemSneddonModel2d(params, use_qpe)
        fem_model.model_setup_and_sol()

        # We save the square root of DOFs in FEM! 
        num_dofs = np.sqrt(len(fem_model.U))
        print("Square root of the number of degrees of freedom:", num_dofs)

        model_sol = FEMComputeSifErrorConvergence(model=fem_model)

        cache = {
            "params": model_sol.params,
            "tip_face_centers": model_sol.tip_face_centers,
            "fracture_node_coords": model_sol.fracture_node_coords,
            "u_jump_normal": model_sol.u_jump_normal,
            "u_jump_tangential": model_sol.u_jump_tangential,
            "num_dofs": num_dofs,
        }
        
        filename  = f"model_{h}_theta{theta_rad}.pkl"
        filepath = os.path.join(save_model, filename)
        with open(filepath, "wb") as dbfile:
            pickle.dump(cache, dbfile)


