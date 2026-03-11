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
from gmshrandom.grids_perturbation import GridsPerturbationMixin
from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from porepy.utils.grid_utils import compute_circumcenter_2d
from fracture import ComputeSifErrorConvergence

import pandas as pd

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.colors import LogNorm

import os
save_fig = "figures/perturb/tpsa/model/diagram"
os.makedirs(save_fig, exist_ok=True)

save_csv = "data/perturb/tpsa/model_diagram"
os.makedirs(save_csv, exist_ok=True)

class TpsaSneddonModel2dPerturbed(
    pp.momentum_balance.TpsaMomentumBalanceMixin,
    GridsPerturbationMixin, 
    ManuSneddonModel2d, 
):
    def __init__(self, params=None):
        super().__init__(params=params)
    
    def set_geometry(self):
        super().set_geometry()
        for sd in self.mdg.subdomains():
            if isinstance(sd, pp.TriangleGrid):
                # Compute circumcenters and add as cell data
                cc, replace = compute_circumcenter_2d(sd)
                sd.cell_centers[:] = cc

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
    "perturb_amplitude": 0.10,
    "perturb_exponent": 1.5,
    "perturb_seed": 1234,
})

model = TpsaSneddonModel2dPerturbed(params=params)
pp.run_time_dependent_model(model)

sd = model.mdg.subdomains(dim=model.nd)[0]
is_acute, num_obtuse = model.check_acute_triangles(sd)
print("Are all triangles acute? ", is_acute)
print("Number of obtuse triangles: ", num_obtuse)


#  Define mesh sizes to convergence analysis
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
# cell_sizes = [0.0125]
seeds = [
    1234, 2029, 31415, 27182, 42424, 
    1414, 3201, 16180, 17333, 22002,
]

diag_results = []

for i, h in enumerate(cell_sizes):
    print(f"=== Running for mesh size h = {h} ===")

    params["meshing_arguments"] = {"cell_size": h}
    params["perturb_seed"] = int(seeds[0])

    model = TpsaSneddonModel2dPerturbed(params=params)
    pp.run_time_dependent_model(model)

    num_dofs = 0
    for g in model.mdg.subdomains(dim=2):
        print("Number of matrix cells:", g.num_cells)
        num_dofs += g.num_cells

    for g in model.mdg.interfaces():
        print("Number of interface cells:", g.num_cells)
        num_dofs += g.num_cells

    num_dofs = np.sqrt(num_dofs)

    compSIF = ComputeSifErrorConvergence(model=model)

    # Normalized cell size
    norm_cellsize = (
        model.params["meshing_arguments"]["cell_size"]
        / model.params["a"]
    )

    pairs_in_cellsizes = compSIF.generate_sampling_pairs()

    error_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    error_std_acc = {tuple(pair): [] for pair in pairs_in_cellsizes}
    r_acc     = {tuple(pair): [] for pair in pairs_in_cellsizes}

    for si, seed in enumerate(seeds):
        print(f"\nRunning seed = {seed}")
        params["perturb_seed"] = int(seed)

        # Run model
        model = TpsaSneddonModel2dPerturbed(params=params)
        pp.run_time_dependent_model(model)

        sd = model.mdg.subdomains(dim=model.nd)[0]
        is_acute, num_obtuse = model.check_acute_triangles(sd)
        print("Are all triangles acute? ", is_acute)
        print("Number of obtuse triangles: ", num_obtuse)
    
        compSIF = ComputeSifErrorConvergence(model=model)
        pairs_in_cellsizes = compSIF.generate_sampling_pairs()
                
        for sampling_pair in pairs_in_cellsizes: 
            compSIF.singularity_order = compSIF.exact_singularity()  
            error_ave, error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
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
diag_data.to_csv(os.path.join(save_csv, "diag_data_so_tpsa.csv"), index=False)
diag_data = pd.read_csv(os.path.join(save_csv, "diag_data_so_tpsa.csv"))

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
    os.path.join(save_fig, "diag_so_mean_tpsa.tiff"),
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
    os.path.join(save_fig, "diag_so_std_tpsa.tiff"),
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
    os.path.join(save_fig, "conv_opt_so_tpsa.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig3)



