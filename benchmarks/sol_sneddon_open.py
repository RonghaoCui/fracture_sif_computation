from __future__ import annotations

import porepy as pp
import numpy as np

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import os
save_dir = "figures/benchmarks//mpsa/sol2"
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
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 45),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.1,
}

model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)

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

# Plot grids 
pp.plot_grid(
    model.mdg,
    rgb=[1, 1, 1],
    fig_size=(3.5,3.5), # inches
    plot_2d=True,           
    linewidth=0.3,
    alpha=0.5,
    title="Grid before fracture propagation: so"
)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "Grids_so_45d.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()

# Plot 2D map of displacement
# Y direction 
for g, data in model.mdg.subdomains(return_data=True):
    # get 'u'（t=0 by default）
    u = data.get(pp.TIME_STEP_SOLUTIONS, {}).get("u", {}).get(0, None)
    if u is not None:
        # get the displacement y along the y direction
        y_disp = u[1::2]
        # plot
        pp.plot_grid(
            g,
            cell_value=y_disp,
            fig_size=(3.5,3.0),  # inches
            plot_2d=True,
            linewidth=0.3,
            title="Y-direction displacement"
        )
    else:
        print(f"Grid {g} has no 'u' defined in TIME_STEP_SOLUTIONS")

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "uy_map_so_45d.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()

# X direction
for g, data in model.mdg.subdomains(return_data=True):
    # get 'u'（t=0 by default）
    u = data.get(pp.TIME_STEP_SOLUTIONS, {}).get("u", {}).get(0, None)
    if u is not None:
        # get the displacement x along the x direction
        x_disp = u[::2]
        # plot
        pp.plot_grid(
            g,
            cell_value=x_disp,
            fig_size=(3.5,3.0),  # inches
            plot_2d=True,
            linewidth=0.3,
            title="X-direction displacement"
        )
    else:
        print(f"Grid {g} has no 'u' defined in TIME_STEP_SOLUTIONS")

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "ux_map_so_45d.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()

# For displacement along the fracture
# Get the 1D fracture grid (assumes only one fracture)
fracture_grid = model.mdg.subdomains(dim=1)[0]
# x_coords = fracture_grid.cell_centers[0]  # X coordinate of each fracture cell center
theta = params["theta_rad"]
t_vec_frac = np.array([np.sin(theta), np.cos(theta)])  # Tangential vector along the fracture
coords = fracture_grid.cell_centers[:2, :]
tip_faces = fracture_grid.tags["tip_faces"].nonzero()[0]
tip_centers = fracture_grid.face_centers[:2, tip_faces]

s_tip = t_vec_frac @ tip_centers
tip0 = tip_centers[:, np.argmin(s_tip)]

rel_coords = coords - tip0[:, None]
x_plot = t_vec_frac @ rel_coords

# Get numerical normal/tangential displacement jump
u_jump = model.displacement_jump([fracture_grid])
u_jump_val = u_jump.value(model.equation_system)
normal_op = model.normal_component([fracture_grid])
u_open_numerical = (normal_op @ u_jump).value(model.equation_system)
tangent_op = model.tangential_component([fracture_grid])
u_slip_numerical = (tangent_op @ u_jump).value(model.equation_system)
u_open_analytical = model.exact_sol.exact_sol_fracture(model.mdg)

# Plot the profile of normal displacement jump along the fracture
fig4, ax4 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax4.plot(
    x_plot,
    u_open_analytical,
    '-',
    color='tab:blue',
    linewidth=2, 
    label="Analytical"
)
ax4.plot(
    x_plot,
    u_open_numerical,
    'o',
    color='tab:orange',
    markersize=4, 
    label="Numerical"
)
ax4.yaxis.set_major_formatter(formatter)
ax4.set_xlabel("Distance from fracture tip 1 ($x$)")
ax4.set_ylabel("Normal displacement jump ($y$)")
ax4.set_title("Normal displacement jump: so")
ax4.set_xlim(0.0, 0.60)
ax4.set_ylim(1.0e-5, 5.0e-5)
ax4.set_xticks(np.arange(0.0, 0.61, 0.1))
ax4.set_yticks(np.arange(1.0e-5, 5.01e-5, 1.0e-5))
ax4.legend()
ax4.grid(True, linestyle="--", alpha=0.5)
fig4.tight_layout()
fig4.savefig(
    os.path.join(save_dir, "un_so_45d.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig4)

# Plot the profile of tangential displacement jump along the fracture
fig5, ax5 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax5.plot(
    x_plot,
    u_slip_numerical,
    '-',
    color='tab:blue',
    linewidth=2, 
    label="Numerical"
)
ax5.yaxis.set_major_formatter(formatter)
ax5.set_xlabel("Distance from fracture tip 1 ($x$)")
ax5.set_ylabel("Tangential displacement jump ($y$)")
ax5.set_title("Tangential displacement jump: so")
ax5.set_xlim(0.0, 0.60)
# ax5.set_ylim(-2.0e-7, 3.0e-7)
ax5.set_xticks(np.arange(0.0, 0.61, 0.1))
# ax5.set_yticks(np.arange(-2.0e-7, 3.01e-7, 1.0e-7))
ax5.legend()
ax5.grid(True, linestyle="--", alpha=0.5)
fig5.tight_layout()
fig5.savefig(
    os.path.join(save_dir, "ut_so_45d.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig5)

