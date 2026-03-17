"""
Convergence analysis for Sneddon problem with auxiliary points using 
multiple seeds of a random number generator. 

"""
from __future__ import annotations

import numpy as np
import porepy as pp
from gmshrandom.auxiliary_points_in_mesh import AuxiliaryPointsMixin
from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fracture import FractureDisplacementJump

import math

import pickle
import os

save_model = "data/mpsa/cache_models/so"
os.makedirs(save_model, exist_ok=True)

class ManuSneddonModel2dWithAuxPoints(AuxiliaryPointsMixin, ManuSneddonModel2d):
    def __init__(self, params=None):
        super().__init__(params=params)

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

# Parameters for auxiliary point generation
params.update({
    "aux_point_count": 32,        # the even number is preferrable
    "aux_point_min_dist": 1.0*params["meshing_arguments"]["cell_size"],  
    "aux_seed": 1234,              
})

model = ManuSneddonModel2dWithAuxPoints(params=params)
pp.run_time_dependent_model(model)

aux_pts = model._aux_points_xy
print("number of points:", aux_pts.shape)

# Save cache models for different mesh sizes and seeds.
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
num_auxiliary_points = [8, 32, 128, 512, 2048, 8192]
seeds = [1234, 2029, 31415, 27182, 42424]
# cell_sizes = [0.1, 0.05]
# num_auxiliary_points = [8, 32]
# seeds = [1234, 2029, 31415, 27182, 42424]

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===")
    params["meshing_arguments"] = {"cell_size": h}
    params["aux_point_count"] = int(num_auxiliary_points[i])
    params["aux_point_min_dist"] = h

    params["aux_seed"] = int(seeds[0])
    model = ManuSneddonModel2dWithAuxPoints(params=params)
    pp.run_time_dependent_model(model)

    model_sol = FractureDisplacementJump(model=model)
    pairs_in_cellsizes = model_sol.generate_sampling_pairs()

    for si, seed in enumerate(seeds):
        print(f"\nRunning seed = {seed}")
        params["aux_seed"] = int(seed)

        # Run model.
        model = ManuSneddonModel2dWithAuxPoints(params=params)
        pp.run_time_dependent_model(model)

        # Calculate total degrees of freedom (DOFs) for the model.
        num_dofs = 0
        for g in model.mdg.subdomains(dim=2):
            print("Number of matrix cells:", g.num_cells)
            num_dofs += g.num_cells

        for g in model.mdg.interfaces():
            print("Number of interface cells:", g.num_cells)
            num_dofs += g.num_cells

        model_sol = FractureDisplacementJump(model=model)

        cache = {
            "params": model_sol.params,
            "frac_cell_centers": model_sol.frac_cell_centers,
            "tip_face_centers": model_sol.tip_face_centers,
            "u_jump_normal": model_sol.u_jump_normal,
            "u_jump_tangential": model_sol.u_jump_tangential,
            "num_dofs": num_dofs,
        }

        filename  = f"model_{h}_seed{seed}.pkl"
        filepath = os.path.join(save_model, filename)
        with open(filepath, "wb") as dbfile:
            pickle.dump(cache, dbfile)

