"""
Evaluate convergence orders of normal displacement jump for Westergaard's mode II problem.

"""

from __future__ import annotations

import porepy as pp
import numpy as np

from westergaard2d import WestergaardShearMode2d

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
save_dir = "figures/benchmarks_conv"
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
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "poi": poi,
    "error_exclusion_zone_fracture_tips": 0.1,
    "tau_inf": 1e-4,    # far_field shear stress
    "mu": 0.0,          # friction coefficient
    "p0": 0.0,          # This pressure value will not be calculated in this model. 
                        # But "p0" is needed because of model's parental class
}

cell_sizes = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.0075, 0.00375]

# Case 1: exclude a small region around fracture tips
error_set1 = []
num_frac_cells_set1 = []

for h in cell_sizes:
    print(f"Running for mesh size h = {h}")
    
    # Update cell size
    params["meshing_arguments"] = {"cell_size": h}

    # Run model
    model = WestergaardShearMode2d(params=params)
    pp.run_time_dependent_model(model)

    # Collect error in tangential displacement
    error = model.collect_data().error_tangential_displacement
    print(f"Relative error: {error:.4e}")
    error_set1.append(error)

    # Get 1D fracture grid (in 2D model)
    fracture_grids = model.mdg.subdomains(dim=1)

    # Usually only one fracture in the problem
    fracture_grid = fracture_grids[0]

    # Number of fracture cells
    num_frac_cells = fracture_grid.num_cells
    print("Number of fracture cells:", num_frac_cells)
    num_frac_cells_set1.append(num_frac_cells)

# Case 2: include a small region around fracture tips
error_set2 = []
num_frac_cells_set2 = []
params["error_exclusion_zone_fracture_tips"] = 0.0

for h in cell_sizes:
    print(f"Running for mesh size h = {h}")
    
    # Update cell size
    params["meshing_arguments"] = {"cell_size": h}

    # Run model
    model = WestergaardShearMode2d(params=params)
    pp.run_time_dependent_model(model)

    # Collect error in tangential displacement
    error = model.collect_data().error_tangential_displacement
    print(f"Relative error: {error:.4e}")
    error_set2.append(error)

    # Get 1D fracture grid (in 2D model)
    fracture_grids = model.mdg.subdomains(dim=1)

    # Usually only one fracture in the problem
    fracture_grid = fracture_grids[0]

    # Number of fracture cells
    num_frac_cells = fracture_grid.num_cells
    print("Number of fracture cells:", num_frac_cells)
    num_frac_cells_set2.append(num_frac_cells)

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
errors_delta_u_set1 = np.array(error_set1)
num_frac_cells_set1 = np.array(num_frac_cells_set1)
errors_delta_u_set2 = np.array(error_set2)
num_frac_cells_set2 = np.array(num_frac_cells_set2)
first_order_line = errors_delta_u_set1[3] * (num_frac_cells_set1[3] / num_frac_cells_set1)

fig, ax = plt.subplots(figsize=(3.5, 3.0))   # inches
ax.loglog(
    num_frac_cells_set1,
    errors_delta_u_set1,
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=4, 
    label="Without tips"
)
ax.loglog(
    num_frac_cells_set2,
    errors_delta_u_set2,
    'o-',
    color='tab:orange',
    linewidth=2,
    markersize=4, 
    label="With tips"
)
ax.plot(
    num_frac_cells_set1,
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
ax.set_ylim(1.0e-3, 1.0)
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
    os.path.join(save_dir, "deltaU_conv_simplex_ws.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)
