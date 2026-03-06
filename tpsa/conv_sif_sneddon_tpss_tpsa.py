"""
Convergence analysis for Westergaard's problem in the shear mode
using the two-point method by TPSA. 

"""

from __future__ import annotations

import porepy as pp
import numpy as np

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fracture import ComputeSifErrorConvergence
from fracture import ComputeSifValue

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/tpsa_results/conv_sif"
os.makedirs(save_dir, exist_ok=True)

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
    "meshing_arguments": {"cell_size": 0.1},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.1,
}


class TpsaSneddonModel2d(
    pp.momentum_balance.TpsaMomentumBalanceMixin, 
    ManuSneddonModel2d,
): 
    pass

# Calculate all SIF values along fracture
h = 0.0125
params["meshing_arguments"] = {"cell_size": h}
model = TpsaSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
compSIF = ComputeSifErrorConvergence(model)

sampling_points = compSIF.generate_samples_on_half_fracture()
u_jump_half_frac, r_cell_to_tip = compSIF.extract_multi_points_normal_jump(sampling_points)
# print("u_jump_half_frac:", u_jump_half_frac)
# print("r_cell_to_tip:", r_cell_to_tip)
SIFValue = ComputeSifValue(model.params)
KIs_half_frac = SIFValue.KI_from_one_point(u_jump_half_frac, r_cell_to_tip, 0.5)
# print("KIs_half_frac:", KIs_half_frac)

# test the two-point option
h = 0.1
params["meshing_arguments"] = {"cell_size": h}
model = TpsaSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
compSIF = ComputeSifErrorConvergence(model)
# Here we use sampling pairs for SIF and this can allow computed singularity to be used
sampling_pair = [0, 2]
# Use the default singularity exponent 0.5
compSIF.singularity_order = compSIF.exact_singularity()  
rel_l2_error_ave, rel_error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)

#  Define mesh sizes to test 
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
# pairs_in_cellsizes = [
#     [2, 3],
#     [4, 5],
#     [7, 8],
#     [11, 12],
#     [20, 21],
#     [33, 34]
# ]
pairs_in_cellsizes = [
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1]
]
error_ave_set = []
error_per_tip_set = []
num_frac_cells_set = []
num_dofs_set = []
for i, h in enumerate(cell_sizes):
    print(f"Running for mesh size h = {h}")
    params["meshing_arguments"] = {"cell_size": h}
    model = TpsaSneddonModel2d(params=params)
    pp.run_time_dependent_model(model)
    compSIF = ComputeSifErrorConvergence(model)
    
    sampling_pair = pairs_in_cellsizes[i]
    compSIF.singularity_order = compSIF.exact_singularity()  
    rel_l2_error_ave, rel_error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
    print("Relative L2 error:", rel_l2_error_ave)
    print("Relative error per tip:", rel_error_per_tip)
    error_ave_set.append(rel_l2_error_ave)
    error_per_tip_set.append(rel_error_per_tip)

    # Number of fracture cells
    num_frac_cells = compSIF.get_number_of_fracture_cells()
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

# Set the plot format
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

# Set the scientific format
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))   

fig2, ax2 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax2.plot(
    r_cell_to_tip[0,:],
    KIs_half_frac[0,:],
    '-',
    color='tab:blue',
    linewidth=2, 
    label="Tip: 1"
)
ax2.plot(
    r_cell_to_tip[1,:],
    KIs_half_frac[1,:],
    '-',
    color='tab:orange',
    linewidth=2, 
    label="Tip: 2"
)
# ax4.xaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)
ax2.set_xlabel("Distance from fracture tip ($r$)")
ax2.set_ylabel("SIF KI")
ax2.set_title("SIF KI: Sneddon (TPSA)")
# ax2.set_xlim(0.0, 0.31)
# ax2.set_ylim(6.5e-5, 10.0e-5)
# ax2.set_xticks(np.arange(0.0, 0.311, 0.05))
# ax2.set_yticks(np.arange(6.5e-5, 10.01e-5, 0.5e-5))
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig(
    os.path.join(save_dir, "TPSA_KIs_rn_so.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()
plt.close(fig2)



# Plot the convergence curve for sif
errors_ave = np.array(error_ave_set)
errors_per_tip = np.array(error_per_tip_set)
num_dofs = np.array(num_dofs_set)
first_order_line = errors_ave[0] * (num_dofs[0] / num_dofs)
half_order_line = errors_ave[0] * (num_dofs[0] / num_dofs) ** 0.5

fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax1.loglog(
    num_dofs,
    errors_per_tip[:, 0],
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Two-point: Tip 1"
)

ax1.loglog(
    num_dofs,
    errors_per_tip[:, 1],
    'o-',
    color='tab:orange',
    linewidth=2,
    markersize=4, 
    label="Two-point: Tip 2"
)

ax1.loglog(
    num_dofs,
    errors_ave,
    'o-',
    color='tab:pink',
    linewidth=2,
    markersize=4, 
    label="Two-point: Average"
)

ax1.plot(
    num_dofs,
    first_order_line,
    '--',
    color='k',
    linewidth=2, 
    label="Reference slope=-1"
)

ax1.plot(
    num_dofs,
    half_order_line,
    '--',
    color='tab:grey',
    linewidth=2, 
    label="Reference slope=-0.5"
)

ax1.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
ax1.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(formatter)
ax1.set_xlim(10, 700)
ax1.set_ylim(0.0002, 1.0)
ax1.xaxis.set_major_locator(LogLocator(base=10))
ax1.xaxis.set_major_formatter(LogFormatterMathtext())
ax1.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax1.yaxis.set_major_locator(LogLocator(base=10))
ax1.yaxis.set_major_formatter(LogFormatterMathtext())
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend(loc="lower left")
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "conv_sif_so_tpss_tpsa.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig1)

print("\n sqrt of dofs: ", num_dofs)
print("\n error of tip 1: ", errors_per_tip[:, 0])
print("\n error of tip 2: ", errors_per_tip[:, 1])
print("\n error of tip ave: ", errors_ave)

