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
from westergaard2d import WestergaardShearMode2d
from fracture import ComputeSifErrorConvergence

import pandas as pd

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext

from matplotlib.colors import LogNorm

import os
save_dir = "figures/mpsa/model/diagram"
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

# Parameters for the exponent function: only for initialization
params.update({
    "constant_c": 0.8,        
    "exponent_alpha": 0.25,              
})

model = WestergaardShearMode2dWithAuxPoints(params=params)
pp.run_time_dependent_model(model)

aux_pts = model._aux_points_xy
print("number of points:", aux_pts.shape)

def compute_cmax(alpha):
    h0 = 0.1
    a = params["a"]
    d_kmax = 0.25
    return d_kmax / ( (a ** (1.0 - alpha)) * (h0 ** alpha) )

exponent_alpha = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
c_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

constant_c_max = {
    alpha: compute_cmax(alpha)
    for alpha in exponent_alpha
}

constant_c_choices = {
    alpha: [ratio * constant_c_max[alpha] for ratio in c_ratios]
    for alpha in exponent_alpha
}

# plt.figure(figsize=(5.0, 3.0))

# for j, ratio in enumerate(c_ratios):
#     alphas = []
#     Cs = []
#     for alpha in exponent_alpha:
#         alphas.append(alpha)
#         Cs.append(constant_c_choices[alpha][j])

#     plt.plot(alphas, Cs, "o-", label=f"C = {ratio} · C_max")

# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$C$")
# plt.yscale("log")  
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# # plt.legend()
# plt.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5), 
#     frameon=False
# )
# plt.tight_layout()
# plt.savefig(
#     os.path.join(save_dir, "choice_c_alpha.tiff"),
#     format="tiff",
#     dpi=300,
#     bbox_inches="tight",
#     pil_kwargs={"compression": "tiff_lzw"}
# )
# plt.show()

#  Define mesh sizes to convergence analysis
# cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
# num_auxiliary_points = [8, 32, 128, 512, 2048, 8192]
# seeds = [1234, 2029, 31415, 27182, 42424]
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
num_auxiliary_points = [8, 32, 128, 512, 2048]
seeds = [1234, 2029, 31415, 27182]

diag_results = []
for alphai in exponent_alpha:
    for ci in constant_c_choices[alphai]:
        params["exponent_alpha"] = alphai
        params["constant_c"] = ci

        err_per_h = []    # list of arrays, each shape (n_seed,)
        dofs_per_h = []

        for i, h in enumerate(cell_sizes):

            error_per_seed = []

            for si, seed in enumerate(seeds):
                print(f"\n=== Running seed = {seed} ===")
                print(f"Running for mesh size h = {h}")
                print(f"Running for alpha = {alphai}")
                print(f"Running for constant_c = {ci}")
    
                params["aux_seed"] = int(seed)
                params["meshing_arguments"] = {"cell_size": h}
                params["aux_point_count"] = int(num_auxiliary_points[i])
                params["aux_point_min_dist"] = h

                # Run model
                model = WestergaardShearMode2dWithAuxPoints(params=params)
                pp.run_time_dependent_model(model)
    
                compSIF = ComputeSifErrorConvergence(model)
                sampling_pair = compSIF.sampling_pairs_by_exponent()

                compSIF.singularity_order = compSIF.exact_singularity()  
                error_ave, error_per_tip = compSIF.two_points_single_scale_tangential(sampling_pair)

                error_per_seed.append(error_ave)
            
            num_dofs = 0
            for g in model.mdg.subdomains(dim=2):
                print("Number of matrix cells:", g.num_cells)
                num_dofs += g.num_cells

            for g in model.mdg.interfaces():
                print("Number of interface cells:", g.num_cells)
                num_dofs += g.num_cells

            num_dofs = np.sqrt(num_dofs)
            print("Square root of the number of degrees of freedom:", num_dofs)
            err_per_h.append(np.array(error_per_seed))
            dofs_per_h.append(num_dofs)

        err_mean = np.array([e.mean() for e in err_per_h])
        err_std  = np.array([e.std(ddof=1) for e in err_per_h])
        dofs = np.array(dofs_per_h)

        x = np.log(dofs)
        y = np.log(err_mean)
        slope, intercept = np.polyfit(x, y, 1)
        
        diag_results.append({
            "alpha": alphai,
            "C": ci,
            "slope": -slope,
        })

diag_data = pd.DataFrame(diag_results)

# R_all   = diag_data["alpha"].values
# DOF_all = diag_data["C"].values
# ERR_all = diag_data["slope"].values

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
    .groupby(["alpha", "C"], as_index=False)
    .agg(mean_error=("slope", "mean"))
)

fig, ax = plt.subplots(figsize=(4.0, 3.0))
sc = ax.scatter(
    df_mean["alpha"],
    df_mean["C"],
    c=df_mean["mean_error"],
    s=28,
    cmap="plasma",     
    linewidths=0,
)

ax.yaxis.set_major_formatter(formatter)
ax.set_xlabel(r"$alpha$")
ax.set_ylabel(r"$C$")
# ax.set_xlim(0.95 * df_mean["r"].min(), 1.05 * df_mean["r"].max())
# ax.set_ylim(0.9  * df_mean["sqrt_dof"].min(),
#             1.1  * df_mean["sqrt_dof"].max())

# colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Mean of convergence orders")

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "diag_c_alpha_ws_scatter_mean.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()


