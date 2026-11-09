"""
Convergence analysis for fracture sliding in the Westergaard mixed-mode benchmark 
by standard quadratic finite element method (QFEM) with fracture rotations using 
multiple fracture angles within the range [0°, 180°). 

This script loads the cache results of the Westergaard mixed-mode benchmark for 
different mesh sizes and fracture orientations, and calculates the map of SIF KII 
errors between radial distances to the tip and mesh sizes.

This script also shows the fitting results of the scaling law between the radial 
distance to the tip and mesh size.  
"""
from __future__ import annotations

import numpy as np
import porepy as pp
from fems import FEMComputeSifErrorConvergence

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.colors import LogNorm

import pickle
import os
save_fig = "figures/sifconv/cache_modes/qfem/wm/map"
os.makedirs(save_fig, exist_ok=True)

save_csv = "data/sifconv/cache_modes/qfem/wm/map"
os.makedirs(save_csv, exist_ok=True)

save_model = "data/sifconv/cache_modes/qfem/wm/models"
os.makedirs(save_model, exist_ok=True)

cell_sizes = [
    0.1, 0.075, 0.05, 0.0375, 0.025, 
    0.01875, 0.0125, 0.009375, 0.00625,
]
theta_rads = [
    0, 15, 30, 45, 60, 75, 
    90, 105, 120, 135, 150, 165,
]

# For averaged error reults. 
diag_results = []
# For error results with different angles.
diag_results_theta = []

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===")

    theta_rad = int(theta_rads[0])
    filename  = f"model_{h}_theta{theta_rad}.pkl"
    filepath = os.path.join(save_model, filename)
    with open(filepath, "rb") as dbfile:
        cache = pickle.load(dbfile)

    num_dofs = np.sqrt(cache["num_dofs"])

    computeSIF = FEMComputeSifErrorConvergence(cache=cache)

    # Normalized cell size
    norm_cellsize = (
        cache["params"]["meshing_arguments"]["cell_size"]
        / cache["params"]["a"]
    )

    pairs_in_cellsizes = computeSIF.generate_sampling_pairs()

    error_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    error_std_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    r_acc     = {tuple(pair): [] for pair in pairs_in_cellsizes}

    for si, theta_rad in enumerate(theta_rads):
        print(f"\nRunning theta_rad = {theta_rad}")

        filename  = f"model_{h}_theta{theta_rad}.pkl"
        filepath = os.path.join(save_model, filename)
        with open(filepath, "rb") as dbfile:
            cache = pickle.load(dbfile)

        computeSIF = FEMComputeSifErrorConvergence(cache=cache)
        pairs_in_cellsizes = computeSIF.generate_sampling_pairs()
                
        for sampling_pair in pairs_in_cellsizes: 
            error_ave, error_per_tip = computeSIF.FEM_two_points_single_scale_tangential(sampling_pair)
            _, r_n = computeSIF.FEM_extract_sampling_tangential_jump(sampling_pair[0])

            r_mean = np.mean(r_n) / cache["params"]["a"]

            # For averaged error results.
            error_acc[tuple(sampling_pair)].append(error_ave)
            error_std_acc[tuple(sampling_pair)].extend(error_per_tip)
            r_acc[tuple(sampling_pair)].append(r_mean)

            # Store error results with each angle. 
            diag_results_theta.append({
                "theta_rad": theta_rad,
                "h": h,
                "r": r_mean,
                "sqrt_dof": num_dofs,
                "norm_h": norm_cellsize,
                "error": error_ave,
                "error_std": np.std(error_per_tip, ddof=0),
            })

    # Store error results by averaging over different angles. 
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
diag_data.to_csv(os.path.join(save_csv, "map_wms.csv"), index=False)
diag_data = pd.read_csv(os.path.join(save_csv, "map_wms.csv"))

diag_data_theta = pd.DataFrame(diag_results_theta)
diag_data_theta.to_csv(os.path.join(save_csv, "map_theta_wms.csv"), index=False)
diag_data_theta = pd.read_csv(os.path.join(save_csv, "map_theta_wms.csv"))

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
    os.path.join(save_fig, "map_wms_mean_qfem.tiff"),
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
    os.path.join(save_fig, "map_wms_std_qfem.tiff"),
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig2)

fit_results_theta = []
best_theta_results = []
for theta_rad in theta_rads:
    df_theta = diag_data_theta[diag_data_theta["theta_rad"] == theta_rad]

    # Get the minimum mean error with correponding h/a and r/a for each angle.
    idx_theta = df_theta.groupby("norm_h")["error"].idxmin()
    df_best_theta = df_theta.loc[idx_theta].reset_index(drop=True)

    best_theta_results.append(df_best_theta)

    # Get the fitting line for the optimal h/a and r/a for each angle.
    x_theta = np.log10(df_best_theta["norm_h"].to_numpy())
    y_theta = np.log10(df_best_theta["r"].to_numpy())

    alpha_theta, logC_theta = np.polyfit(x_theta, y_theta, 1)
    C_theta = 10 ** logC_theta

    fit_results_theta.append({
        "theta_rad": theta_rad,
        "alpha": alpha_theta,
        "C": C_theta,
    })

fit_data_theta = pd.DataFrame(fit_results_theta)
fit_data_theta.to_csv(os.path.join(save_csv, "fit_data_theta_wms.csv"), index=False)
fit_data_theta = pd.read_csv(os.path.join(save_csv, "fit_data_theta_wms.csv"))

# Statistics of fitting results with different angles. 
alpha_mean_theta = fit_data_theta["alpha"].mean()
alpha_std_theta = fit_data_theta["alpha"].std(ddof=0)
C_mean_theta = fit_data_theta["C"].mean()
C_std_theta = fit_data_theta["C"].std(ddof=0)

print("=== Statistics of fitting results by different angles ===")
print("Mean alpha:", alpha_mean_theta)
print("Std alpha:", alpha_std_theta)
print("Mean C:", C_mean_theta)
print("Std C:", C_std_theta)

df_best_all_theta = pd.concat(best_theta_results, ignore_index=True)
df_band = (
    df_best_all_theta
    .groupby("norm_h", as_index=False)
    .agg(
        r_mean=("r", "mean"),
        r_std=("r", "std"),
        r_min=("r", "min"),
        r_max=("r", "max"),
    )
    .sort_values("norm_h")
    .reset_index(drop=True)
)

df_band["r_upper"] = df_band["r_mean"] + df_band["r_std"]
df_band["r_lower"] = df_band["r_mean"] - df_band["r_std"]

x_band = np.log10(df_band["norm_h"].to_numpy())
y_band = np.log10(df_band["r_mean"].to_numpy())

alpha_band, logC_band = np.polyfit(x_band, y_band, 1)
C_band = 10 ** logC_band
print("The alpha value using mean of optimal r/a:", alpha_band)
print("The C value using mean of optimal r/a:", C_band)

# Get the minimum mean error with correponding h/a and r/a
idx = df_mean.groupby("norm_h")["mean_error"].idxmin()
df_best = df_mean.loc[idx].reset_index(drop=True)

# Get the fitting line for the optimal h/a and r/a
x = np.log10(df_best["norm_h"].to_numpy())
y = np.log10(df_best["r"].to_numpy())

print("x values for fitting:", x)
print("y values for fitting:", y)

alpha, logC = np.polyfit(x, y, 1)
C = 10 ** logC

print("The alpha value using averaged error map:", alpha)
print("The C value using averaged error map:", C)

norm_h_fit = np.linspace(0.001, 1.0, 999,)
r_fit = C * norm_h_fit ** alpha
r_fit_theta = C_mean_theta * norm_h_fit ** alpha_mean_theta
r_fit_band = C_band * norm_h_fit ** alpha_band

# Plot the curves of h/a and r/a with the minimum error
fig3, ax3 = plt.subplots(figsize=(3.5, 3.0))
ax3.loglog(
    df_band["norm_h"],
    df_band["r_mean"],
    'o',
    color='tab:orange',
    # linewidth=2,
    markersize=5,
    label="Mean optimal $r/a$"
)

ax3.loglog(
    norm_h_fit,
    r_fit_band,
    '--',
    color='tab:orange',
    linewidth=2,
    label="Fitting (Mean optimal $r/a$)"
)

ax3.fill_between(
    df_band["norm_h"],
    df_band["r_lower"],
    df_band["r_upper"],
    color='tab:orange',
    alpha=0.35,
    linewidth=0.0,
    label=r"$\pm 1$ STD of optimal $r/a$"
)

# Plot the fitting line using averaged maps.
ax3.loglog(
    df_best["norm_h"],
    df_best["r"],
    'o',
    color='tab:blue',
    # linewidth=2,
    markersize=5, 
    label="Data (averaged map)"
)

ax3.loglog(
    norm_h_fit,
    r_fit,
    '--',
    color='tab:blue',
    linewidth=2,
    label="Fitting (averaged map)"
)

ax3.set_xlabel(r"$h/a$")
ax3.set_ylabel(r"$r/a$")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax3.yaxis.set_major_formatter(formatter)
ax3.set_xlim(0.008, 1.0)
ax3.set_ylim(0.07, 1.0)
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
    os.path.join(save_fig, "scaling_law_wms_qfem.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig3)
