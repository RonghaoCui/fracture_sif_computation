"""
Evaluate the singularity order (exponent) at fracture tips within a simulation domain for Sneddon's problem.

Currently, the script only supports a single fracture with two fracture tips.

Features: 
 - Provide two options of selecting fracture cell locations (cell centers) along the fracture for evaluating
   singularity order (exponent): 

        Single-scale option: evaluate singularity at the same grid refinement level by two cell centers (i and j)
        near the fracture tip. It is noted that cells can be flexibly selected. We use the following as an 
        example: the cell center i is selected at the fracture tip cell (r_i=h/2), while the cell center j can be 
        movable to the third cell closest to the frature tip (r_j=5*h/2).

        Two-scale option: cell centers i and j, whose distance are r_i and r_j, are movable and selected from fine and 
        coarse grids, respectively. If the refinement levels of coarse and fine grids are k_c and k_f, where k_f=k_c+1, 
        the distance from cell center j to the fracture tip face in coarse grids is (4*k_c+1)*h_c/2 and the distance from 
        cell center j in fine grids is (4*k_f+1)*h_f/2. 

 - Perform both single-scale and two-scale convergence curve analysis. 
   The single-scale analysis evaluates convergence only with respect to mesh size (cell size: h),
   while the two-scale analysis involves coarse and fine models.

* Note: This script now is only valid for the opening model (KI), but will be potentially extended to 
        cover the SIF in the shear mode (KII). 

"""
from __future__ import annotations

import porepy as pp
import numpy as np

import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fracture import FractureDisplacementJump
from fracture import ComputeSingularityErrorForModel

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/conv_sing"
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

# Chose the sampling option for fracture cells 
# params.update({
#     "sampling": {"mode": "refinement", "level": 1}
# })

# params.update({
#     "sampling": {"mode": "fixed", "steps": 1}
# })

# test the single-scale option for convergence error at one step
h = 0.1
params["meshing_arguments"] = {"cell_size": h}
# params["sampling"] = {"mode": "fixed", "steps": 1}
model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
comp = ComputeSingularityErrorForModel(model)
sampling_pair = [0, 2]
rel_l2_error_ave, rel_error_per_tip = comp.single_scale_option_normal(sampling_pair)
print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)

# test the two-scale option for convergence error at one step
h_coarse = 0.1
refinement_level = 1
# params["meshing_arguments"] = {"cell_size": h_coarse}
# params["sampling"] = {"mode": "refinement", "level": refinement_level}
model_coarse = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model_coarse)
comp_coarse = ComputeSingularityErrorForModel(model_coarse)

h_fine = h_coarse / 2.0
params["meshing_arguments"] = {"cell_size": h_fine}
# params["sampling"] = {"mode": "refinement", "level": refinement_level+1}
model_fine = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model_fine)
jumpmodel_fine = FractureDisplacementJump(model_fine)
sampling_pair = [refinement_level*2, (refinement_level+1)*2]
rel_l2_error_ave, rel_error_per_tip = comp_coarse.two_scale_option_normal(jumpmodel_fine, sampling_pair)

print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)

# Convergence curves of the single-scale option
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
pairs_in_cellsizes = [
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2]
]
error_ave_set = []
error_per_tip_set = []
num_frac_cells_set = []
for i, h in enumerate(cell_sizes):
    params["meshing_arguments"] = {"cell_size": h}
    # this steps value is only for keyword initialization 
    # and does not work for the simulation results 
    # params["sampling"] = {"mode": "fixed", "level": 1}
    model = ManuSneddonModel2d(params=params)
    pp.run_time_dependent_model(model)
    comp = ComputeSingularityErrorForModel(model)
    # This steps_pair will work for the keywords: params["sampling"] = {"mode": "fixed", "steps": X}
    sampling_pair = pairs_in_cellsizes[i]
    rel_l2_error_ave, rel_error_per_tip = comp.single_scale_option_normal(sampling_pair)
    print("Relative L2 error:", rel_l2_error_ave)
    print("Relative error per tip:", rel_error_per_tip)
    error_ave_set.append(rel_l2_error_ave)
    error_per_tip_set.append(rel_error_per_tip)

    # Number of fracture cells
    num_frac_cells = comp.get_number_of_fracture_cells()
    print("Number of fracture cells:", num_frac_cells)
    num_frac_cells_set.append(num_frac_cells)

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

# Plot the convergence curve for singularity exponents
errors_ave = np.array(error_ave_set)
errors_per_tip = np.array(error_per_tip_set)
num_frac_cells = np.array(num_frac_cells_set)
first_order_line = errors_ave[3] * (num_frac_cells[3] / num_frac_cells)

fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax1.loglog(
    num_frac_cells,
    errors_per_tip[:, 0],
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Single-scale: Tip 1"
)

ax1.loglog(
    num_frac_cells,
    errors_per_tip[:, 1],
    'o-',
    color='tab:orange',
    linewidth=2,
    markersize=4, 
    label="Single-scale: Tip 2"
)

ax1.loglog(
    num_frac_cells,
    errors_ave,
    'o-',
    color='tab:pink',
    linewidth=2,
    markersize=4, 
    label="Single-scale: Average"
)

ax1.plot(
    num_frac_cells,
    first_order_line,
    '--',
    color='k',
    linewidth=2, 
    label="Reference slope=-1"
)

ax1.set_xlabel("Number of fracture cells")
ax1.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(formatter)
ax1.set_xlim(3.0, 220)
ax1.set_ylim(1.0e-2, 1.0)
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
    os.path.join(save_dir, "conv_sing_so_single.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig1)

# Convergence curves of the two-scale option
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
refinement_level = 1
error_ave_set = []
error_per_tip_set = []
num_frac_cells_set = []
for i, h in enumerate(cell_sizes):
    h_coarse = h
    params["meshing_arguments"] = {"cell_size": h_coarse}
    # params["sampling"] = {"mode": "refinement", "level": refinement_level}
    model_coarse = ManuSneddonModel2d(params=params)
    pp.run_time_dependent_model(model_coarse)
    comp_coarse = ComputeSingularityErrorForModel(model_coarse)

    h_fine = h_coarse / 2.0
    params["meshing_arguments"] = {"cell_size": h_fine}
    # params["sampling"] = {"mode": "refinement", "level": refinement_level+1}
    model_fine = ManuSneddonModel2d(params=params)
    pp.run_time_dependent_model(model_fine)
    jumpmodel_fine = FractureDisplacementJump(model_fine)

    sampling_pair = [(refinement_level+1)*2, refinement_level*2]
    rel_l2_error_ave, rel_error_per_tip = comp_coarse.two_scale_option_normal(jumpmodel_fine, sampling_pair)

    print("Relative L2 error:", rel_l2_error_ave)
    print("Relative error per tip:", rel_error_per_tip)
    error_ave_set.append(rel_l2_error_ave)
    error_per_tip_set.append(rel_error_per_tip)

    # Number of fracture cells by use of the coarse model
    num_frac_cells = comp_coarse.get_number_of_fracture_cells()
    print("Number of fracture cells:", num_frac_cells)
    num_frac_cells_set.append(num_frac_cells)

    refinement_level = refinement_level + 1

# Plot the convergence curve for singularity exponents
errors_ave = np.array(error_ave_set)
errors_per_tip = np.array(error_per_tip_set)
num_frac_cells = np.array(num_frac_cells_set)
first_order_line = errors_ave[0] * (num_frac_cells[0] / num_frac_cells)

fig2, ax2 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax2.loglog(
    num_frac_cells,
    errors_per_tip[:, 0],
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Two-scale: Tip 1"
)

ax2.loglog(
    num_frac_cells,
    errors_per_tip[:, 1],
    'o-',
    color='tab:orange',
    linewidth=2,
    markersize=4, 
    label="Two-scale: Tip 2"
)

ax2.loglog(
    num_frac_cells,
    errors_ave,
    'o-',
    color='tab:pink',
    linewidth=2,
    markersize=4, 
    label="Two-scale: Average"
)

ax2.plot(
    num_frac_cells,
    first_order_line,
    '--',
    color='k',
    linewidth=2, 
    label="Reference slope=-1"
)

ax2.set_xlabel("Number of fracture cells")
ax2.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax2.yaxis.set_major_formatter(formatter)
ax2.set_xlim(3.0, 220)
ax2.set_ylim(1.0e-2, 1.0)
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_major_formatter(LogFormatterMathtext())
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax2.yaxis.set_major_locator(LogLocator(base=10))
ax2.yaxis.set_major_formatter(LogFormatterMathtext())
ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend(loc="lower left")
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "conv_sing_so_two.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig2)