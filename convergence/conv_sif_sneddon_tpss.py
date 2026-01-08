

from __future__ import annotations

import porepy as pp
import numpy as np

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fracture import ComputeSifErrorConvergence

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/conv_sif"
os.makedirs(save_dir, exist_ok=True)

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
    "meshing_arguments": {"cell_size": 0.1},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.1,
}

# test the two-point option
h = 0.1
params["meshing_arguments"] = {"cell_size": h}
model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
compSIF = ComputeSifErrorConvergence(model)
# Here we use sampling pairs for SIF and this can allow computed singularity to be used
sampling_pair = [0, 2]
# Use the default singularity exponent 0.5
compSIF.singularity_order = compSIF.exact_singularity()  
rel_l2_error_ave, rel_error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)