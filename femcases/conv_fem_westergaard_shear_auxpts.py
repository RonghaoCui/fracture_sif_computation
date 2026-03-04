"""
Calculate convergence orders of SIF KI in Sneddon's problem by FEM with QPEs, 
using two-point methods.

"""
from __future__ import annotations

import porepy as pp
import numpy as np

from fems import FemQpeWestergaardShearMode2dWithAuxPts
from fems import FEMComputeSifErrorConvergence

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext

import os
save_dir = "figures/fem_results/conv"
os.makedirs(save_dir, exist_ok=True)

a = 0.30
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

fem_model = FemQpeWestergaardShearMode2dWithAuxPts(params)
fem_model.model_setup_and_sol()

# Only for the horizontal fracture
x_coords, u_jump_normal, u_jump_tangential = fem_model.extract_displacement_jump()
#  Define mesh sizes to convergence analysis
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
num_auxiliary_points = [8, 32, 128, 512, 2048]
seeds = [1234, 2029, 31415, 27182, 42424]

# exponent = 0.25
# pairs_in_cellsizes = [
#     [5, 7],
#     [7, 9],
#     [11, 13],
#     [17, 19],
#     [27, 29]
# ]

pairs_in_cellsizes = [
    [2, 3],
    [3, 4],
    [5, 6],
    [9, 10],
    [14, 15]
]

all_err_ave = []        # list of (n_h,) arrays
all_err_per_tip = []    # list of (n_h, 2) arrays
all_dofs = []           # list of (n_h,) arrays
for si, seed in enumerate(seeds):
    error_ave_set = []
    error_per_tip_set = []
    num_frac_cells_set = []
    num_dofs_set = []

    for i, h in enumerate(cell_sizes):
        print(f"\n=== Running seed = {seed} ===")
        print(f"Running for mesh size h = {h}")
    
        params["aux_seed"] = int(seed)
        params["meshing_arguments"] = {"cell_size": h}
        params["aux_point_count"] = int(num_auxiliary_points[i])
        params["aux_point_min_dist"] = h

        # Run model
        fem_model = FemQpeWestergaardShearMode2dWithAuxPts(params=params)
        fem_model.model_setup_and_sol()
    
        computeSIF = FEMComputeSifErrorConvergence(fem_model)
    
        sampling_pair = pairs_in_cellsizes[i]
        rel_l2_error_ave, rel_error_per_tip = computeSIF.FEM_two_points_single_scale_tangential(sampling_pair)

        error_ave_set.append(rel_l2_error_ave)
        error_per_tip_set.append(rel_error_per_tip)

        num_dofs = np.sqrt(len(fem_model.U))
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
    os.path.join(save_dir, "FEM_conv_sif_ws_random_tpss.tiff"),
    # os.path.join(save_dir, "FEM_conv_sif_so_random_withcoeff.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)

print("\n sqrt of dofs: ", mean_dofs)
print("\n mean_err_ave: ", mean_err_ave)
