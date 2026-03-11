"""
An example for Sneddon problem with auxiliary points. 

"""
from __future__ import annotations

import numpy as np
import porepy as pp
from gmshrandom.grids_perturbation import GridsPerturbationMixin
from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

import os
save_dir = "figures/perturb/mpsa/model/pert_grids"
os.makedirs(save_dir, exist_ok=True)

class ManuSneddonModel2dPerturbed(
    GridsPerturbationMixin, 
    ManuSneddonModel2d
):
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

# Parameters for grid perturbation
params.update({
    "perturb_amplitude": 0.25,
    "perturb_exponent": 1.5,
    "perturb_seed": 1234,
})

model = ManuSneddonModel2dPerturbed(params=params)
pp.run_time_dependent_model(model)

sd = model.mdg.subdomains(dim=model.nd)[0]
is_acute, num_obtuse = model.check_acute_triangles(sd)
print("Are all triangles acute? ", is_acute)
print("Number of obtuse triangles: ", num_obtuse)

# Plot grids only
mpl.rcParams.update({
    "font.family": "serif",     
    "font.size": 8,             
    "axes.labelsize": 8,       
    "axes.titlesize": 8, 
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0, 
    "lines.markersize": 2.0, 
    "mathtext.fontset": "dejavusans", 
})
formatter = ScalarFormatter(useMathText=True)
pp.plot_grid(
    model.mdg,
    rgb=[1, 1, 1],
    fig_size=(3.5,3.5), # inches
    plot_2d=True,           
    linewidth=0.3,
    alpha=0.5,
    title="Perturbed grid: so"
)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "Grids_so_perturbed.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()

