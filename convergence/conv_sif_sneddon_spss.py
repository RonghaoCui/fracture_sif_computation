"""
Evaluate the stress intensity factor (SIF) in the opening mode (KI) at fracture tips 
within a simulation domain for Sneddon's problem.

Currently, the script only supports a single fracture with two fracture tips.

Features: 
 - Use fracture cell locations (cell centers) along the fracture from two options (single-scale and two-scale)
   for evaluating the SIF in single-point and two-point ways. 

- Calculate   
    SIF KI: 
        Single-scale single-point option: The location is selected from one fracture cell center in the scale 
        of radial distance. The example on the third cell closest to the frature tip (r_j=5*h/2) is provided. 
        The exact singularity order is set as 0.5 by default.

        Single-scale two-point option: The location is selected from two fracture cell centers in the 
        single-scale option. The example on the third cell closest to the frature tip (r_j=5*h/2) is provided. 
        The singularity order is set as 0.5 by default, or comes from the single-scale option. 
            
        Two-scale two-point option: The locations are selected on two cell centers from the two mesh. The SIF 
        expression is derived from theoretical solution with elimination of the second term. The singularity 
        order is also from the two-scale option (Option 3a). But the exact singularity order (0.5 by default) 
        can be allowed as Option 3b. 

 - Perform both single-point and two-point convergence analysis. 
   The convergence analysis on SIF can be conducted not only with respect to mesh size (cell size: h),
   but also the two-scale analysis involves coarse and fine models.

* Note: This script now is only valid for the opening model (KI), but will be potentially extended to 
        cover the SIF in the shear mode (KII).  

"""

from __future__ import annotations

import porepy as pp
import numpy as np

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fracture import ComputeSifErrorConvergence

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/conv_sif"
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

# test the single-point option
h = 0.1
params["meshing_arguments"] = {"cell_size": h}
model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
compSIF = ComputeSifErrorConvergence(model)
# Here we use sampling pairs instead of single point, because this can allow computed singularity to be used
# But for SIF, we only use the second element
sampling_pair = [0, 2]
# Use the default singularity exponent 0.5
compSIF.singularity_order = compSIF.exact_singularity()  
rel_l2_error_ave, rel_error_per_tip = compSIF.single_point_single_scale_normal(sampling_pair)
print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)

#  Define mesh sizes to test 
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
pairs_in_cellsizes = [
    [0, 2],
    [0, 4],
    [0, 6],
    [0, 10],
    [0, 15],
    [0, 25]
]
error_ave_set = []
error_per_tip_set = []
num_frac_cells_set = []
for i, h in enumerate(cell_sizes):
    print(f"Running for mesh size h = {h}")
    params["meshing_arguments"] = {"cell_size": h}
    model = ManuSneddonModel2d(params=params)
    pp.run_time_dependent_model(model)
    compSIF = ComputeSifErrorConvergence(model)
    
    sampling_pair = pairs_in_cellsizes[i]
    compSIF.singularity_order = compSIF.exact_singularity()  
    rel_l2_error_ave, rel_error_per_tip = compSIF.single_point_single_scale_normal(sampling_pair)
    print("Relative L2 error:", rel_l2_error_ave)
    print("Relative error per tip:", rel_error_per_tip)
    error_ave_set.append(rel_l2_error_ave)
    error_per_tip_set.append(rel_error_per_tip)

    # Number of fracture cells
    num_frac_cells = compSIF.get_number_of_fracture_cells()
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
first_order_line = errors_ave[0] * (num_frac_cells[0] / num_frac_cells)
half_order_line = errors_ave[0] * (num_frac_cells[0] / num_frac_cells) ** 0.5

fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax1.loglog(
    num_frac_cells,
    errors_per_tip[:, 0],
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Single-point: Tip 1"
)

ax1.loglog(
    num_frac_cells,
    errors_per_tip[:, 1],
    'o-',
    color='tab:orange',
    linewidth=2,
    markersize=4, 
    label="Single-point: Tip 2"
)

ax1.loglog(
    num_frac_cells,
    errors_ave,
    'o-',
    color='tab:pink',
    linewidth=2,
    markersize=4, 
    label="Single-point: Average"
)

ax1.plot(
    num_frac_cells,
    first_order_line,
    '--',
    color='k',
    linewidth=2, 
    label="Reference slope=-1"
)

ax1.plot(
    num_frac_cells,
    half_order_line,
    '--',
    color='tab:grey',
    linewidth=2, 
    label="Reference slope=-0.5"
)

ax1.set_xlabel("Number of fracture cells")
ax1.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(formatter)
ax1.set_xlim(5.5, 220)
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
    os.path.join(save_dir, "conv_sif_so_spss.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig1)


# test the two-point option
h = 0.1
params["meshing_arguments"] = {"cell_size": h}
model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
compSIF = ComputeSifErrorConvergence(model)
# Here we use sampling pairs for SIF and this can allow computed singularity to be used
sampling_pair = [0, 2]
# Use the default singularity exponent 0.5
compSIF.singularity_order = compSIF.exact_singularity()  
rel_l2_error_ave, rel_error_per_tip = compSIF.two_points_single_scale_normal(sampling_pair)
print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)

