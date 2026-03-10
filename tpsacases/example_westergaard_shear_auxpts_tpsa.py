"""
An example for Westargaard problem in the shear mode with auxiliary points. 

"""
from __future__ import annotations

# import os, sys
# print("cwd =", os.getcwd())
# print("sys.path[0] =", sys.path[0])

import numpy as np
import porepy as pp
from gmshrandom.auxiliary_points_in_mesh import AuxiliaryPointsMixin
from westergaard2d import WestergaardShearMode2d

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

import os
save_dir = "figures/tpsa/model/aux_grids"
os.makedirs(save_dir, exist_ok=True)

class TpsaWestergaardShearMode2dWithAuxPts(
    pp.momentum_balance.TpsaMomentumBalanceMixin,
    AuxiliaryPointsMixin, 
    WestergaardShearMode2d, 
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
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.0,
    "tau_inf": 1e-4,    # far_field shear stress
    "mu": 0.0,          # friction coefficient
    "p0": 0.0,          # This pressure value will not be calculated in this model. 
                        # But "p0" is needed because of model's parental class. 
}

# Parameters for auxiliary point generation
params.update({
    "aux_point_count": 32,        # the even number is preferrable
    "aux_point_min_dist": 1.0*params["meshing_arguments"]["cell_size"],  
    "aux_seed": 1234,              
})

model = TpsaWestergaardShearMode2dWithAuxPts(params=params)
pp.run_time_dependent_model(model)

aux_pts = model._aux_points_xy
print("number of points:", aux_pts.shape)

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
    title="Grid (tpsa): ws"
)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "Grids_ws_tpsa.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()

# plot grids and auxiliary points together
FIG_NUM = 777   
fig2 = plt.figure(FIG_NUM, figsize=(3.5,3.5), clear=True) 
pp.plot_grid(
    model.mdg,
    rgb=[1, 1, 1],
    plot_2d=True,
    linewidth=0.3,
    alpha=1.0,
    title=None,
    fig_num=FIG_NUM,   
    if_plot=False,    
)

ax2 = fig2.axes[0]   
ax2.scatter(
    aux_pts[:, 0], aux_pts[:, 1],
    s=10, marker='o',
    facecolors='r', edgecolors='r', 
    zorder=8, label='Aux points'
)
plt.savefig(
    os.path.join(save_dir, "Grids_ws_aux_tpsa.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()