"""
Evaluate the stress intensity factor (SIF) in the opening mode (KI) at fracture tips 
within a simulation domain for Sneddon's problem.

Currently, the script only supports a single fracture with two fracture tips.

Features: 
 - Use fracture cell locations (cell centers) along the fracture from two options (single-scale and two-scale)
   for evaluating the SIF in single-point and two-point ways. 

- Calculate   
    SIF KI: 
        Single-scale single-point option:
            Option 1: The location is selected from one of two fracture cell centers in the scale of radial distance. 
            The example on the third cell closest to the frature tip (r_j=5*h/2) is provided. The exact singularity 
            order is set as 0.5 by default.
            Option 2: The location is selected from one of two fracture cell centers in the single-scale option. 
            The example on the third cell closest to the frature tip (r_j=5*h/2) is provided. The singularity order 
            is from the single-scale option.
            Option 3: The location is selected on cell center j from the two-scale option. The singularity order is 
            also from the two-scale option (Option 3a). But the exact singularity order (0.5 by default) can be 
            allowed as Option 3b. 

        Two-point option: the expression is derived from theoretical solution with elimination of the second term. 

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
from fracture import FractureDisplacementJump
from fracture import ComputeSingularityErrorForModel
from fracture import ComputeSifErrorConvergence

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "convergence"
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

params.update({
    "sampling": {"mode": "fixed", "steps": 1}
})

# test the single-point option
h = 0.1
params["meshing_arguments"] = {"cell_size": h}
params["sampling"] = {"mode": "fixed", "steps": 1}
model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)
compSIF = ComputeSifErrorConvergence(model)
steps_pair=(0, 2)
compSIF.singularity_order = compSIF.exact_singularity()
rel_l2_error_ave, rel_error_per_tip = compSIF.single_point_single_scale_normal(steps_pair)
print("Relative L2 error:", rel_l2_error_ave)
print("Relative error per tip:", rel_error_per_tip)

#  Define mesh sizes to test 
cell_sizes = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.0075, 0.00375]
error_set = []
num_frac_cells_set = []

# Run simulations 
for h in cell_sizes:
    print(f"Running for mesh size h = {h}")
    
    # Update cell size
    params["meshing_arguments"] = {"cell_size": h}

    # Run model
    model = ManuSneddonModel2d(params=params)
    pp.run_time_dependent_model(model)

    # Collect error in normal displacement
    error = model.collect_data().error_displacement
    print(f"Relative error: {error:.4e}")
    error_set.append(error)

    # Get 1D fracture grid (in 2D model)
    fracture_grids = model.mdg.subdomains(dim=1)

    # Usually only one fracture in Sneddon problem
    fracture_grid = fracture_grids[0]

    # Number of fracture cells
    num_frac_cells = fracture_grid.num_cells
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

# Plot the convergence curve for displacement jump
errors_delta_u = np.array(error_set)
num_frac_cells = np.array(num_frac_cells_set)
first_order_line = errors_delta_u[3] * (num_frac_cells[3] / num_frac_cells)

fig, ax = plt.subplots(figsize=(3.5, 3.0))   # inches
ax.loglog(
    num_frac_cells,
    errors_delta_u,
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Numerical"
)
ax.plot(
    num_frac_cells,
    first_order_line,
    '--',
    color='k',
    linewidth=2, 
    label="Reference slope=-1"
)

ax.set_xlabel("Number of fracture cells")
ax.set_ylabel("Relative error")

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter)
ax.set_xlim(1.0, 220)
ax.set_ylim(1.0e-4, 1.0e-1)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "deltaU_converg_simplex_so.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)
