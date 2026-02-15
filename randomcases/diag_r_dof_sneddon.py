"""
Convergence analysis for Sneddon problem with auxiliary points using 
multiple seeds of a random number generator. 

"""
from __future__ import annotations

# import os, sys
# print("cwd =", os.getcwd())
# print("sys.path[0] =", sys.path[0])

import numpy as np
import porepy as pp
from gmshrandom.auxiliary_points_in_mesh import AuxiliaryPointsMixin
from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fracture import ComputeSifErrorConvergence

import pandas as pd
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.colors import LogNorm

import os
save_dir = "figures/diagram"
os.makedirs(save_dir, exist_ok=True)

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


#  Define mesh sizes to convergence analysis
# cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
# num_auxiliary_points = [8, 32, 128, 512, 2048, 8192]
# seeds = [1234, 2029, 31415, 27182, 42424]

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

cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
num_auxiliary_points = [8, 32, 128, 512, 2048, 8192]
# cell_sizes = [0.1, 0.05]
# num_auxiliary_points = [8, 32]
seeds = [1234, 2029, 31415, 27182, 42424]

diag_results = []

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===")
    params["meshing_arguments"] = {"cell_size": h}
    params["aux_point_count"] = int(num_auxiliary_points[i])
    params["aux_point_min_dist"] = h

    params["aux_seed"] = int(seeds[0])
    model = ManuSneddonModel2dWithAuxPoints(params=params)
    pp.run_time_dependent_model(model)

    num_dofs = 0
    for g in model.mdg.subdomains(dim=2):
        print("Number of matrix cells:", g.num_cells)
        num_dofs += g.num_cells

    for g in model.mdg.interfaces():
        print("Number of interface cells:", g.num_cells)
        num_dofs += g.num_cells

    num_dofs = np.sqrt(num_dofs)

    compSIF = ComputeSifErrorConvergence(model)
    pairs_in_cellsizes = compSIF.generate_sampling_pairs()

    error_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    error_std_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    r_acc     = {tuple(pair): [] for pair in pairs_in_cellsizes}

    for si, seed in enumerate(seeds):
        print(f"\nRunning seed = {seed}")
        params["aux_seed"] = int(seed)

        # Run model
        model = ManuSneddonModel2dWithAuxPoints(params=params)
        pp.run_time_dependent_model(model)
    
        compSIF = ComputeSifErrorConvergence(model)
        pairs_in_cellsizes = compSIF.generate_sampling_pairs()
                
        for sampling_pair in pairs_in_cellsizes: 
            compSIF.singularity_order = compSIF.exact_singularity()  
            error_ave, error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
            _, r_n = compSIF.extract_sampling_normal_jump(sampling_pair[0])

            r_mean = np.mean(r_n)

            error_acc[tuple(sampling_pair)].append(error_ave)
            error_std_acc[tuple(sampling_pair)].extend(error_per_tip)
            r_acc[tuple(sampling_pair)].append(r_mean)
     
    for sampling_pair in pairs_in_cellsizes:
        pair_key = tuple(sampling_pair)
        errs_mean = np.array(error_acc[pair_key])
        errs_std  = np.array(error_std_acc[pair_key])

        diag_results.append({
            "h": h,
            "r": np.mean(r_acc[pair_key]),
            "sqrt_dof": num_dofs,
            "error": errs_mean.mean(),
            "error_std": errs_std.std(ddof=0),
        })

diag_data = pd.DataFrame(diag_results)

R_all   = diag_data["r"].values
DOF_all = diag_data["sqrt_dof"].values
ERR_all = diag_data["error"].values
ERR_std_all = diag_data["error_std"].values

# print("R_all =", R_all)
# print("DOF_all =", DOF_all)
# print("ERR_all =", ERR_all)

# plot
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

df_mean = (
    diag_data
    .groupby(["sqrt_dof", "r"], as_index=False)
    .agg(mean_error=("error", "mean"))
)
vmin = df_mean["mean_error"].min()
vmax = df_mean["mean_error"].max()

fig, ax = plt.subplots(figsize=(4.0, 3.0))
sc = ax.scatter(
    df_mean["r"],
    df_mean["sqrt_dof"],
    c=df_mean["mean_error"],
    s=28,
    cmap="plasma",     
    linewidths=0,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)

ax.yaxis.set_major_formatter(formatter)
ax.set_yscale("log")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\sqrt{\mathrm{DOF}}$")
ax.set_xlim(0.95 * df_mean["r"].min(), 1.05 * df_mean["r"].max())
ax.set_ylim(0.9  * df_mean["sqrt_dof"].min(),
            1.1  * df_mean["sqrt_dof"].max())

# colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Mean of relative error")

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "diag_r_dof_so_scatter_mean.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()

# Plot std
df_std = (
    diag_data
    .groupby(["sqrt_dof", "r"], as_index=False)
    .agg(std_error=("error_std", "mean"))
)

vmin_std = df_std["std_error"].min()
vmax_std = df_std["std_error"].max()

fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))

sc2 = ax2.scatter(
    df_std["r"],
    df_std["sqrt_dof"],
    c=df_std["std_error"],
    s=28,
    cmap="plasma", 
    linewidths=0,
    norm=LogNorm(vmin=vmin_std, vmax=vmax_std),
)

ax2.set_yscale("log")
ax2.set_xlabel(r"$r$")
ax2.set_ylabel(r"$\sqrt{\mathrm{DOF}}$")
ax2.set_xlim(0.95 * df_std["r"].min(), 1.05 * df_std["r"].max())
ax2.set_ylim(
    0.9 * df_std["sqrt_dof"].min(),
    1.1 * df_std["sqrt_dof"].max()
)

cbar = plt.colorbar(sc2, ax=ax2)
cbar.set_label("STD of relative error")

ax2.yaxis.set_major_locator(LogLocator(base=10))
ax2.yaxis.set_major_formatter(LogFormatterMathtext())
ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax2.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "diag_r_dof_so_scatter_std.tiff"),
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()

# bins
# unique_dof = np.unique(DOF_all)

# dof_bins = np.zeros(len(unique_dof) + 1)
# dof_bins[1:-1] = 0.5 * (unique_dof[:-1] + unique_dof[1:])
# dof_bins[0] = unique_dof[0] * 0.95
# dof_bins[-1] = unique_dof[-1] * 1.05

# r_bins = np.linspace(R_all.min(), R_all.max(), 20)

# stat_mean, r_edges, dof_edges, _ = binned_statistic_2d(
#     R_all, DOF_all, ERR_all,
#     statistic="mean",
#     bins=[r_bins, dof_bins]
# )

# # plot
# mpl.rcParams.update({
#     "font.family": "serif",     
#     "font.size": 8,             
#     "axes.labelsize": 8,       
#     "axes.titlesize": 8, 
#     "legend.fontsize": 7,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "lines.linewidth": 1.0, 
#     "lines.markersize": 2.0, 
#     "mathtext.fontset": "dejavusans", 
# })
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_powerlimits((0, 0))            # Scientific form

# fig, ax = plt.subplots(figsize=(4.0, 3.0))
# pcm = ax.pcolormesh(
#     r_edges,
#     dof_edges,
#     stat_mean.T,
#     shading="auto",
#     cmap="plasma"
# )

# ax.set_xlim(0.95 * R_all.min(), 1.05 * R_all.max())
# ax.set_ylim(0.95 * DOF_all.min(), 1.05 * DOF_all.max())

# # ax.set_xscale("log")
# ax.set_yscale("log")

# ax.set_xlabel(r"$r$")
# ax.set_ylabel(r"$\sqrt{\mathrm{DOF}}$")

# cbar = plt.colorbar(pcm, ax=ax)
# cbar.set_label("Mean relative SIF error")

# fig, ax = plt.subplots(figsize=(4.0, 3.5))

# for dof in np.unique(DOF_all):
#     mask = DOF_all == dof
#     r_vals = R_all[mask]
#     err_vals = ERR_all[mask]

#     if len(r_vals) < 3:
#         continue

#     r_bins = np.linspace(r_vals.min(), r_vals.max(), 20)

#     stat, edges, _ = binned_statistic(
#         r_vals,
#         err_vals,
#         statistic="mean",
#         bins=r_bins
#     )

#     y0 = dof_bins[j]
#     y1 = dof_bins[j + 1]

#     ax.pcolormesh(
#         r_edges,
#         [y0, y1],          
#         stat[np.newaxis, :],
#         cmap="plasma",
#         norm=norm,
#         shading="auto"
#     )

# ax.set_xlabel(r"$r$")
# ax.set_ylabel("Mean relative SIF error")
# ax.set_yscale("log")

# ax.legend(frameon=False, fontsize=7)

# ax.grid(True, which="both", linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.savefig(
#     os.path.join(save_dir, "diag_r_dof_sneddon_bin.tiff"),
#     format="tiff",
#     dpi=300,
#     bbox_inches="tight",
#     pil_kwargs={"compression": "tiff_lzw"}
# )
# plt.show()


