"""
Convergence analysis for the Westergaard opening-mode benchmark by MPSA with 
fracture rotations using multiple fracture angles within the range [0°, 180°). 

This script runs the Westergaard opening-mode benchmark for different mesh 
sizes and fracture orientations, and saves the results in a cache for later 
analysis.
"""
from __future__ import annotations

import numpy as np
import porepy as pp
from westergaard2d import WestergaardOpenMode2d
from fracture import FractureDisplacementJump

import math

import pickle
import os

save_model = "data/sifconv/cache_modes/mpsa/wo/models"
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
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.0,
    "sigma_inf": 1e-4,  # far_field normal stress
    "mu": 0.0,          # friction coefficient
    "p0": 0.0,          # This pressure value will not be calculated in this model. 
                        # But "p0" is needed because of model's parental class
}

model = WestergaardOpenMode2d(params=params)
pp.run_time_dependent_model(model)

print("=== Initial model run completed. ===")

# Save cache models for different mesh sizes and seeds.
# cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
cell_sizes = [
    0.1, 0.075, 0.05, 0.0375, 0.025, 
    0.01875, 0.0125, 0.009375, 0.00625, 0.0046875, 0.003125,
]
theta_rads = [
    0, 15, 30, 45, 60, 75, 
    90, 105, 120, 135, 150, 165,
]

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===") 
    params["meshing_arguments"] = {"cell_size": h}

    model = WestergaardOpenMode2d(params=params)
    pp.run_time_dependent_model(model)

    model_sol = FractureDisplacementJump(model=model)
    pairs_in_cellsizes = model_sol.generate_sampling_pairs()

    for si, theta_rad in enumerate(theta_rads):
        print(f"\nRunning theta_rad = {theta_rad}")
        params["theta_rad"] = math.radians(90 - theta_rad)
        
        # Run model.
        model = WestergaardOpenMode2d(params=params)
        pp.run_time_dependent_model(model)

        # Calculate total degrees of freedom (DOFs) for the model.
        # Only DOFs in the matrix are considered.
        num_dofs = 0
        for g in model.mdg.subdomains(dim=2):
            print("Number of matrix cells:", g.num_cells)
            num_dofs += g.num_cells

        # for g in model.mdg.interfaces():
        #     print("Number of interface cells:", g.num_cells)
        #     num_dofs += g.num_cells

        model_sol = FractureDisplacementJump(model=model)

        cache = {
            "params": model_sol.params,
            "frac_cell_centers": model_sol.frac_cell_centers,
            "tip_face_centers": model_sol.tip_face_centers,
            "u_jump_normal": model_sol.u_jump_normal,
            "u_jump_tangential": model_sol.u_jump_tangential,
            "num_dofs": num_dofs,
        }

        filename  = f"model_{h}_theta{theta_rad}.pkl"
        filepath = os.path.join(save_model, filename)
        with open(filepath, "wb") as dbfile:
            pickle.dump(cache, dbfile)

