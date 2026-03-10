"""
Calculate convergence orders of SIF KI in Sneddon's problem by FEM with QPEs, 
using two-point methods.

"""
from __future__ import annotations

import porepy as pp
import numpy as np

from fems import FEMWestergaardShearMode2d
from fems import FEMComputeSifErrorConvergence

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/femqpe/model/conv"
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

fem_model = FEMWestergaardShearMode2d(params)
fem_model.model_setup_and_sol()

# Only for the horizontal fracture
x_coords, u_jump_normal, u_jump_tangential = fem_model.extract_displacement_jump()

computeSIF = FEMComputeSifErrorConvergence(fem_model)
sampling_pair = [2, 3]
rel_l2_error_ave, rel_error_per_tip = computeSIF.FEM_two_points_single_scale_tangential(sampling_pair)
print("Relative L2 error 1:", rel_l2_error_ave)
print("Relative error per tip 1:", rel_error_per_tip)

sampling_pair = [0, 1, 2, 3]
u_jump, r_cell_to_tip = computeSIF.FEM_extract_multi_points_tangential(sampling_pair)
print("u_jump:", u_jump)
print("r_cell_to_tip:", r_cell_to_tip)

# sampling_pair = [2, 3]
# rel_l2_error_ave, rel_error_per_tip = computeSIF.FEM_multi_points_single_scale_normal(sampling_pair)
# print("Relative L2 error 2:", rel_l2_error_ave)
# print("Relative error per tip 2:", rel_error_per_tip)

xL = fem_model.length/2.0 - fem_model.a
count_r = np.sum((x_coords - xL) <= fem_model.a) 
interval_r = np.arange(count_r)
interval_r = interval_r.tolist()
u_jump_t, r_t = computeSIF.FEM_extract_multi_points_tangential(interval_r)
_ , KIIs_t = computeSIF.FEM_KII_from_multi_points(u_jump_t, r_t, 0.5)

# Force to set the scientific form
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))

fig4, ax4 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax4.plot(
    r_t[0,:],
    KIIs_t[0,:],
    '-',
    color='tab:blue',
    linewidth=2, 
    label="Tip: 1"
)
ax4.plot(
    r_t[1,:],
    KIIs_t[1,:],
    '-',
    color='tab:orange',
    linewidth=2, 
    label="Tip: 2"
)
# ax4.xaxis.set_major_formatter(formatter)
ax4.yaxis.set_major_formatter(formatter)
ax4.set_xlabel("Distance from fracture tip ($r$)")
ax4.set_ylabel("SIF KII")
ax4.set_title("SIF KII: Westergaard (FEM)")
ax4.set_xlim(0.0, 0.31)
ax4.set_ylim(6.5e-5, 10.0e-5)
ax4.set_xticks(np.arange(0.0, 0.311, 0.05))
ax4.set_yticks(np.arange(6.5e-5, 10.01e-5, 0.5e-5))
ax4.legend()
ax4.grid(True, linestyle="--", alpha=0.5)
fig4.tight_layout()
fig4.savefig(
    os.path.join(save_dir, "FEM_KIIs_rt_ws.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()
plt.close(fig4)

#  Define mesh sizes to test 
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
# pairs_in_cellsizes = [
#     [2, 3],
#     [4, 5],
#     [7, 8],
#     [11, 12],
#     [20, 21]
# ]
# pairs_in_cellsizes = [
#     [5, 7],
#     [9, 11],
#     [15, 17],
#     [23, 25],
#     [41, 43]
# ]
pairs_in_cellsizes = [
    [5, 7],
    [7, 9],
    [11, 13],
    [17, 19],
    [27, 29]
]
error_ave_set = []
error_per_tip_set = []
num_dofs_set = []
for i, h in enumerate(cell_sizes):
    print(f"Running for mesh size h = {h}")
    params["meshing_arguments"] = {"cell_size": h}
    fem_model = FEMWestergaardShearMode2d(params)
    fem_model.model_setup_and_sol()
    computeSIF = FEMComputeSifErrorConvergence(fem_model)

    sampling_pair = pairs_in_cellsizes[i]
     
    rel_l2_error_ave, rel_error_per_tip = computeSIF.FEM_two_points_single_scale_tangential(sampling_pair)
    print("Relative L2 error:", rel_l2_error_ave)
    print("Relative error per tip:", rel_error_per_tip)
    error_ave_set.append(rel_l2_error_ave)
    error_per_tip_set.append(rel_error_per_tip)

    # Square root of number of DOFs
    num_dofs = np.sqrt(len(fem_model.U))
    print("Square root of the number of degrees of freedom:", num_dofs)
    num_dofs_set.append(num_dofs)

# Set the scientific format
logformat = ScalarFormatter(useMathText=True)
logformat.set_powerlimits((0, 0))   

# Plot the convergence curve 
errors_ave = np.array(error_ave_set)
errors_per_tip = np.array(error_per_tip_set)
num_sqrt_dofs = np.array(num_dofs_set)

# Reference line
num_dofs_line = np.array([10, 30, 50, 70, 90, 100, 300, 500, 1000])
initial_value_line = 4e-3
first_order_line = initial_value_line * (num_dofs_line[0] / num_dofs_line)
half_order_line = initial_value_line  * (num_dofs_line[0] / num_dofs_line) ** 0.5

fig5, ax5 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax5.loglog(
    num_sqrt_dofs,
    errors_per_tip[:, 0],
    'o-',
    color='tab:blue',
    linewidth=1.5,
    markersize=3, 
    label="Two-point: Tip 1"
)

ax5.loglog(
    num_sqrt_dofs,
    errors_per_tip[:, 1],
    'o-',
    color='tab:orange',
    linewidth=1.5,
    markersize=3, 
    label="Two-point: Tip 2"
)

ax5.loglog(
    num_sqrt_dofs,
    errors_ave,
    'o-',
    color='tab:pink',
    linewidth=1.5,
    markersize=3, 
    label="Two-point: Average"
)

ax5.plot(
    num_dofs_line,
    first_order_line,
    '--',
    color='k',
    linewidth=1.5, 
    label="Reference slope=-1"
)

ax5.plot(
    num_dofs_line,
    half_order_line,
    '--',
    color='tab:grey',
    linewidth=1.5, 
    label="Reference slope=-0.5"
)

ax5.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
ax5.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax5.xaxis.set_major_formatter(logformat)
ax5.yaxis.set_major_formatter(logformat)
ax5.set_xlim(10, 700)
ax5.set_ylim(0.0002, 1.0)
ax5.xaxis.set_major_locator(LogLocator(base=10))
ax5.xaxis.set_major_formatter(LogFormatterMathtext())
ax5.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax5.yaxis.set_major_locator(LogLocator(base=10))
ax5.yaxis.set_major_formatter(LogFormatterMathtext())
ax5.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax5.grid(True, which="both", linestyle="--", alpha=0.5)
ax5.legend(loc="lower left")
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "FEM_conv_sif_ws_tpss.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()
plt.close(fig5)

