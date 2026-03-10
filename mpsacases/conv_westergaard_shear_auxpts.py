"""
Convergence analysis for Westergaard problem in the shear mode with 
auxiliary points using multiple seeds of a random number generator. 

"""
from __future__ import annotations

# import os, sys
# print("cwd =", os.getcwd())
# print("sys.path[0] =", sys.path[0])

import numpy as np
import porepy as pp
from gmshrandom.auxiliary_points_in_mesh import AuxiliaryPointsMixin
from westergaard2d import WestergaardShearMode2d
from fracture import ComputeSifErrorConvergence

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext

import os
save_dir = "figures/aux_sif"
os.makedirs(save_dir, exist_ok=True)

class WestergaardShearMode2dWithAuxPoints(AuxiliaryPointsMixin, WestergaardShearMode2d):
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
                        # But "p0" is needed because of model's parental class
}

# Parameters for auxiliary point generation
params.update({
    "aux_point_count": 32,        # the even number is preferrable
    "aux_point_min_dist": 1.0*params["meshing_arguments"]["cell_size"],  
    "aux_seed": 1234,              
})

model = WestergaardShearMode2dWithAuxPoints(params=params)
pp.run_time_dependent_model(model)

aux_pts = model._aux_points_xy
print("number of points:", aux_pts.shape)

#  Define mesh sizes to convergence analysis
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
num_auxiliary_points = [8, 32, 128, 512, 2048, 8192]
seeds = [1234, 2029, 31415, 27182, 42424]

# We employ an exponent-type function for point selections
# exponent = 0.10
# k_targets = [0, 0, 0, 0, 0, 0]
# exponent = 0.25
# k_targets = [2, 4, 7, 11, 20, 33]
# exponent = 0.5
# k_targets = [2, 3, 5, 7, 10, 14]
# exponent = 0.4
# k_targets = [2, 3, 5, 8, 13, 20]
# exponent = 0.3
# k_targets = [2, 4, 6, 10, 17, 28]
# exponent = 0.2
# k_targets = [2, 4, 7, 13, 22, 40]
# exponent = 0.1
# k_targets = [2, 4, 8, 16, 30, 56]
# exponent = 0.6
# k_targets = [2, 3, 4, 5, 7, 10]

# cell_sizes = [0.1, 0.05]
# # k_targets = [2, 4]
# num_auxiliary_points = [8, 32]
# seeds = [1234]

# exponent = 0.25
# pairs_in_cellsizes = [
#     [2, 3],
#     [4, 5],
#     [7, 8],
#     [11, 12],
#     [20, 21],
#     [33, 34]
# ]

pairs_in_cellsizes = [
    [2, 3],
    [3, 4],
    [6, 7],
    [11, 12],
    [19, 20],
    [32, 33]
]

all_err_ave = []        # list of (n_h,) arrays
all_err_per_tip = []    # list of (n_h, 2) arrays
all_dofs = []           # list of (n_h,) arrays
for si, seed in enumerate(seeds):
    error_ave_set = []
    error_per_tip_set = []
    num_frac_cells_set = []
    num_dofs_set = []
    
    # exponent = 0.10
    # k_targets = [0, 0, 0, 0, 0, 0]
    # exponent = 0.25
    # k_targets = [2, 4, 7, 11, 20, 33]
    # k_targets = [2, 4]

    for i, h in enumerate(cell_sizes):
        print(f"\n=== Running seed = {seed} ===")
        print(f"Running for mesh size h = {h}")
    
        params["aux_seed"] = int(seed)
        params["meshing_arguments"] = {"cell_size": h}
        params["aux_point_count"] = int(num_auxiliary_points[i])
        params["aux_point_min_dist"] = h

        # Run model
        model = WestergaardShearMode2dWithAuxPoints(params=params)
        pp.run_time_dependent_model(model)
    
        compSIF = ComputeSifErrorConvergence(model)
    
        sampling_pair = pairs_in_cellsizes[i]
        compSIF.singularity_order = compSIF.exact_singularity()  
        error_ave, error_per_tip = compSIF.two_points_single_scale_tangential(sampling_pair)

        error_ave_set.append(error_ave)
        error_per_tip_set.append(error_per_tip)

        # Get 1D fracture grid (in 2D model)
        fracture_grids = model.mdg.subdomains(dim=1)
        # Usually only one fracture in Sneddon problem
        fracture_grid = fracture_grids[0]
        # Number of fracture cells
        num_frac_cells = fracture_grid.num_cells
        print("Number of fracture cells:", num_frac_cells)
        num_frac_cells_set.append(num_frac_cells)

        num_dofs = 0
        for g in model.mdg.subdomains(dim=2):
            print("Number of matrix cells:", g.num_cells)
            num_dofs += g.num_cells

        for g in model.mdg.interfaces():
            print("Number of interface cells:", g.num_cells)
            num_dofs += g.num_cells

        num_dofs = np.sqrt(num_dofs)
        print("Square root of the number of degrees of freedom:", num_dofs)
        num_dofs_set.append(num_dofs)

    all_err_ave.append(np.array(error_ave_set))           # shape: (n_h,)
    all_err_per_tip.append(np.vstack(error_per_tip_set))  # shape: (n_h, 2)
    all_dofs.append(np.array(num_dofs_set))               # shape: (n_h,)

err_ave_mat = np.vstack(all_err_ave)                          # (n_seed, n_h)
err_tip1_mat = np.vstack([E[:, 0] for E in all_err_per_tip])  # (n_seed, n_h)
err_tip2_mat = np.vstack([E[:, 1] for E in all_err_per_tip])  # (n_seed, n_h)
dofs_mat = np.vstack(all_dofs)                                # (n_seed, n_h)

mean_err_ave = err_ave_mat.mean(axis=0)
std_err_ave  = err_ave_mat.std(axis=0)

mean_tip1 = err_tip1_mat.mean(axis=0)
std_tip1  = err_tip1_mat.std(axis=0)

mean_tip2 = err_tip2_mat.mean(axis=0)
std_tip2  = err_tip2_mat.std(axis=0)

# 
mean_dofs = dofs_mat.mean(axis=0)
print("Dofs:", mean_dofs)

# Reference line
num_dofs_line = np.array([10, 30, 50, 70, 90, 100, 300, 500, 1000])
initial_value_line = 1e-3
first_order_line = initial_value_line * (num_dofs_line[0] / num_dofs_line)
half_order_line = initial_value_line  * (num_dofs_line[0] / num_dofs_line) ** 0.5

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
formatter.set_powerlimits((0, 0))            # Scientific form

fig, ax = plt.subplots(figsize=(5.0, 3.0))   # inches
for si in range(len(seeds)):
    ax.loglog(
        all_dofs[si], all_err_ave[si],
        'o-', color='tab:blue',
        linewidth=2.0,
        markersize=4,
        alpha=0.35, 
        label=f"Seed {si+1}"
    )

ax.loglog(mean_dofs, mean_err_ave, 
          'd-', color='tab:blue', 
          linewidth=2.0, 
          markersize=4,
          label="Average over 5 seeds"
)

ax.loglog(
    num_dofs_line, first_order_line,
    '--', color='k',
    linewidth=2.0,
    label="Reference slope=-1"
)

# ax.loglog(
#     num_dofs_line, half_order_line,
#     '--', color='tab:grey',
#     linewidth=2.0,
#     label="Reference slope=-0.5"
# )

# ax.set_xlabel("Number of fracture cells")
ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
ax.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter)
# ax.set_xlim(5.5, 220)
# ax.set_ylim(0.0002, 1.0)
ax.set_xlim(10, 700)
ax.set_ylim(0.0002, 1.0)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.grid(True, which="both", linestyle="--", alpha=0.5)
# ax.legend(loc="lower left")
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  
    frameon=False
)
plt.tight_layout()
plt.savefig(
    # os.path.join(save_dir, "conv_sif_ws_random_tpss.tiff"),
    os.path.join(save_dir, "conv_sif_ws_with_coeff.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)

print("\n sqrt of dofs: ", mean_dofs)
print("\n mean_err_ave: ", mean_err_ave)

