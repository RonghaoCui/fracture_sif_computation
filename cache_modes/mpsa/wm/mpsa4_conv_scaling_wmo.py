"""
Convergence analysis for KI in the Westergaard mixed-mode benchmark by MPSA with 
fracture rotations using multiple fracture angles within the range [0°, 180°). 

This script loads the cache results of the Westergaard mixed-mode benchmark for 
different mesh sizes and fracture orientations, and calculates SIF KI errors with 
the mesh refinement, using the scaling-based sampling strategy.
"""
from __future__ import annotations

import numpy as np
import porepy as pp
from fracture import ComputeSifErrorConvergence

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext

import pickle
import os
save_model = "data/sifconv/cache_modes/mpsa/wm/models"
os.makedirs(save_model, exist_ok=True)

save_fig = "figures/sifconv/cache_modes/mpsa/wm/conv"
os.makedirs(save_fig, exist_ok=True)

save_csv = "data/sifconv/cache_modes/mpsa/wm/conv"
os.makedirs(save_csv, exist_ok=True)

#  Define mesh sizes to convergence analysis
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
# cell_sizes = [
#     0.1, 0.075, 0.05, 0.0375, 0.025, 
#     0.01875, 0.0125, 0.009375, 0.00625, 0.0046875, 0.003125,
# ]
theta_rads = [
    0, 15, 30, 45, 60, 75, 
    90, 105, 120, 135, 150, 165,
]

# We employ an exponent-type function for point selections
C = 0.714
alpha = 0.298

all_err_ave = []        # list of (n_h,) arrays
all_err_per_tip = []    # list of (n_h, 2) arrays
all_dofs = []           # list of (n_h,) arrays
for si, theta_rad in enumerate(theta_rads):
    error_ave_set = []
    error_per_tip_set = []
    num_frac_cells_set = []
    num_dofs_set = []

    for i, h in enumerate(cell_sizes):
        print(f"=== Running for frac angle theta = {theta_rad} ===")
        print(f"Running for mesh size h = {h}")
    
        filename  = f"model_{h}_theta{theta_rad}.pkl"
        filepath = os.path.join(save_model, filename)
        with open(filepath, "rb") as dbfile:
            cache = pickle.load(dbfile)

        num_dofs = np.sqrt(cache["num_dofs"])

        compSIF = ComputeSifErrorConvergence(cache=cache)
    
        # sampling_pair = pairs_in_cellsizes[i]
        sampling_pair = compSIF.sampling_pairs_from_power_function(C, alpha)
        compSIF.singularity_order = compSIF.exact_singularity()  
        error_ave, error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)

        error_ave_set.append(error_ave)
        error_per_tip_set.append(error_per_tip)

        # Number of fracture cells
        num_frac_cells = compSIF.frac_num_cells
        print("Number of fracture cells:", num_frac_cells)
        num_frac_cells_set.append(num_frac_cells)

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
for si in range(len(theta_rads)):
    ax.loglog(
        all_dofs[si], all_err_ave[si],
        'o-', color='tab:blue',
        linewidth=2.0,
        markersize=4,
        alpha=0.35, 
        label=f"Angle {si+1}"
    )

ax.loglog(mean_dofs, mean_err_ave, 
          'd-', color='tab:blue', 
          linewidth=2.0, 
          markersize=4,
          label="Average over 12 angles"
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
    os.path.join(save_fig, "conv_sif_wmo_mpsa_scaling.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)

# For saving results. 
conv_results = []
for i, h in enumerate(cell_sizes):
    conv_results.append({
        "sqrt_dof": mean_dofs[i],
        "error_ave": mean_err_ave[i],
        "std_error_ave": std_err_ave[i],
    })

conv_data = pd.DataFrame(conv_results)
conv_data.to_csv(os.path.join(save_csv, "conv_sif_wmo_mpsa_scaling.csv"), index=False)

print("\n sqrt_dofs: ", mean_dofs)
print("\n error_ave: ", mean_err_ave)
print("\n std_error_ave: ", std_err_ave)

