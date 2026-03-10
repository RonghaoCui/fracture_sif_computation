"""
Convergence analysis for Westergaard shear problem with auxiliary points using 
multiple seeds of a random number generator. 

"""
from __future__ import annotations

import numpy as np
import porepy as pp
from fems import FemQpeWestergaardShearMode2dWithAuxPts
from fems import FEMComputeSifErrorConvergence

import math

import pickle
import os

save_model = "data/femqpe/cache_models/ws"
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
    "p0": 0.0,
    "tau_inf": 1e-4,    # far_field shear stress
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.0,
}

# Parameters for auxiliary point generation
params.update({
    "aux_point_count": 32,        # the even number is preferrable
    "aux_point_min_dist": 1.0*params["meshing_arguments"]["cell_size"],  
    "aux_seed": 1234,              
})

fem_model = FemQpeWestergaardShearMode2dWithAuxPts(params=params)
fem_model.model_setup_and_sol()


# Save cache models for different mesh sizes and seeds.

cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
num_auxiliary_points = [8, 32, 128, 512, 2048]
# cell_sizes = [0.1, 0.05]
# num_auxiliary_points = [8, 32]
seeds = [1234, 2029, 31415, 27182, 42424]
# seeds = [1234, 2029]

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===")
    params["meshing_arguments"] = {"cell_size": h}
    params["aux_point_count"] = int(num_auxiliary_points[i])
    params["aux_point_min_dist"] = h

    params["aux_seed"] = int(seeds[0])
    fem_model = FemQpeWestergaardShearMode2dWithAuxPts(params=params)
    fem_model.model_setup_and_sol()

    model_sol = FEMComputeSifErrorConvergence(model=fem_model)

    for si, seed in enumerate(seeds):
        print(f"\nRunning seed = {seed}")
        params["aux_seed"] = int(seed)

        # Run model
        fem_model = FemQpeWestergaardShearMode2dWithAuxPts(params=params)
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

        filename  = f"model_{h}_seed{seed}.pkl"
        filepath = os.path.join(save_model, filename)
        with open(filepath, "wb") as dbfile:
            pickle.dump(cache, dbfile)


