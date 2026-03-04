"""
Convergence analysis for Sneddon problem with auxiliary points using 
multiple seeds of a random number generator by TPSA. 

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
save_fig = "figures/tpsa_results/diagram"
os.makedirs(save_fig, exist_ok=True)

save_csv = "data/tpsa/diagram"
os.makedirs(save_csv, exist_ok=True)

class TpsaSneddonModel2dWithAuxPoints(
    pp.momentum_balance.TpsaMomentumBalanceMixin,
    AuxiliaryPointsMixin, 
    ManuSneddonModel2d, 
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

# Parameters for auxiliary point generation
params.update({
    "aux_point_count": 32,        # the even number is preferrable
    "aux_point_min_dist": 1.0*params["meshing_arguments"]["cell_size"],  
    "aux_seed": 1234,              
})

model = TpsaSneddonModel2dWithAuxPoints(params=params)
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
    model = TpsaSneddonModel2dWithAuxPoints(params=params)
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

    # Number of fracture cells
    # num_frac_cells = compSIF.get_number_of_fracture_cells() 
    # print("Number of fracture cells:", num_frac_cells)
    # Normalized cell size
    norm_cellsize = (
        model.params["meshing_arguments"]["cell_size"]
        / model.params["a"]
    )

    # pairs_in_cellsizes = compSIF.generate_sampling_pairs()
    pairs_in_cellsizes = compSIF.generate_samples_on_half_fracture()
    pairs_in_cellsizes = pairs_in_cellsizes[1:]

    error_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    error_std_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    r_acc     = {tuple(pair): [] for pair in pairs_in_cellsizes}

    for si, seed in enumerate(seeds):
        print(f"\nRunning seed = {seed}")
        params["aux_seed"] = int(seed)

        # Run model
        model = TpsaSneddonModel2dWithAuxPoints(params=params)
        pp.run_time_dependent_model(model)
    
        compSIF = ComputeSifErrorConvergence(model)
        # pairs_in_cellsizes = compSIF.generate_sampling_pairs()
        # pairs_in_cellsizes = compSIF.generate_sampling_three()
                
        for sampling_pair in pairs_in_cellsizes: 
            compSIF.singularity_order = compSIF.exact_singularity()  
            # error_ave, error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
            error_ave, error_per_tip = compSIF.multipoints_single_scale_normal(int(sampling_pair[0]), 1)
            _, r_n = compSIF.extract_sampling_normal_jump(sampling_pair[0])

            r_mean = np.mean(r_n) / model.params["a"]

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
            "norm_h": norm_cellsize,
            "error": errs_mean.mean(),
            "error_std": errs_std.std(ddof=0),
        })

diag_data = pd.DataFrame(diag_results)
diag_data.to_csv(os.path.join(save_csv, "diag_data_so_tpsa_multi.csv"), index=False)
diag_data = pd.read_csv(os.path.join(save_csv, "diag_data_so_tpsa_multi.csv"))

# R_all   = diag_data["r"].values
# DOF_all = diag_data["sqrt_dof"].values
# ERR_all = diag_data["error"].values
# ERR_std_all = diag_data["error_std"].values

# print("R_all =", R_all)
# print("DOF_all =", DOF_all)
# print("ERR_all =", ERR_all)

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

# Plot mean
df_mean = (
    diag_data
    .groupby(["norm_h", "r"], as_index=False)
    .agg(mean_error=("error", "mean"))
)

vmin = df_mean["mean_error"].min()
vmax = df_mean["mean_error"].max()

fig1, ax1 = plt.subplots(figsize=(4.0, 3.0))
sc = ax1.scatter(
    df_mean["r"],
    df_mean["norm_h"],
    c=df_mean["mean_error"],
    s=28,
    cmap="plasma",     
    linewidths=0,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)

ax1.yaxis.set_major_formatter(formatter)
ax1.set_yscale("log")
ax1.set_xlabel(r"$r/a$")
# ax1.set_ylabel(r"$\sqrt{\mathrm{DOF}}$")
ax1.set_ylabel(r"$h/a$")
# ax1.set_xlim(0.95 * df_mean["r"].min(), 1.05 * df_mean["r"].max())
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0.9  * df_mean["norm_h"].min(),
             1.1  * df_mean["norm_h"].max())

# colorbar
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label("Mean of relative error")

ax1.yaxis.set_major_locator(LogLocator(base=10))
ax1.yaxis.set_major_formatter(LogFormatterMathtext())
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(save_fig, "diag_r_dof_so_mean_tpsa_multi.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig1)

# Plot std
df_std = (
    diag_data
    .groupby(["norm_h", "r"], as_index=False)
    .agg(std_error=("error_std", "mean"))
)

vmin_std = df_std["std_error"].min()
vmax_std = df_std["std_error"].max()

fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))

sc2 = ax2.scatter(
    df_std["r"],
    df_std["norm_h"],
    c=df_std["std_error"],
    s=28,
    cmap="plasma", 
    linewidths=0,
    norm=LogNorm(vmin=vmin_std, vmax=vmax_std),
)

ax2.set_yscale("log")
ax2.set_xlabel(r"$r/a$")
# ax2.set_ylabel(r"$\sqrt{\mathrm{DOF}}$")
ax2.set_ylabel(r"$h/a$")
# ax2.set_xlim(0.95 * df_std["r"].min(), 1.05 * df_std["r"].max())
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(
    0.9 * df_std["norm_h"].min(),
    1.1 * df_std["norm_h"].max()
)

cbar = plt.colorbar(sc2, ax=ax2)
cbar.set_label("STD of relative error")

ax2.yaxis.set_major_locator(LogLocator(base=10))
ax2.yaxis.set_major_formatter(LogFormatterMathtext())
ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax2.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(save_fig, "diag_r_dof_so_std_tpsa_multi.tiff"),
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig2)

# Get the minimum mean error with correponding sqrt_dof and r
idx = df_mean.groupby("norm_h")["mean_error"].idxmin()
df_best = df_mean.loc[idx].reset_index(drop=True)

# Get the fitting line for the optimal sqrt_dof and r
x = np.log10(df_best["norm_h"].to_numpy())
y = np.log10(df_best["r"].to_numpy())

alpha, logC = np.polyfit(x, y, 1)
C = 10 ** logC

print("The alpha value:", alpha)
print("The C value", C)

norm_h_fit = np.linspace(0.0, 1.0, 100,)
r_fit = C * norm_h_fit ** alpha

# Plot the curves of sqrt_dof and r with the minimum error
fig3, ax3 = plt.subplots(figsize=(3.5, 3.0))
ax3.loglog(
    df_best["norm_h"],
    df_best["r"],
    'o',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Data"
)

ax3.loglog(
    norm_h_fit,
    r_fit,
    '--',
    color='tab:orange',
    linewidth=2,
    label="Fitting"
)

ax3.set_xlabel(r"$r/a$")
ax3.set_ylabel(r"$h/a$")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax3.yaxis.set_major_formatter(formatter)
# ax3.set_xlim(10, 700)
# ax3.set_ylim(0.0002, 1.0)
ax3.set_xlim(0.01, 1.0)
ax3.set_ylim(0.1, 1.0)
ax3.xaxis.set_major_locator(LogLocator(base=10))
ax3.xaxis.set_major_formatter(LogFormatterMathtext())
ax3.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax3.yaxis.set_major_locator(LogLocator(base=10))
ax3.yaxis.set_major_formatter(LogFormatterMathtext())
ax3.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax3.grid(True, which="both", linestyle="--", alpha=0.5)
ax3.legend(loc="lower left")
plt.tight_layout()
plt.savefig(
    os.path.join(save_fig, "conv_r_dof_opt_so_tpsa_multi.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig3)



