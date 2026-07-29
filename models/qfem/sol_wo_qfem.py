"""Generate visualization figures for the 2D Westergaard opening-mode benchmark using 
the solver of standard quadratic finite element method (QFEM). The computational geometry 
is meshed based on PorePy.

Quarter-point elements (QPEs) are disabled based on the `use_qpe` flag.

This script:
    1. Runs the 2D benchmark with a simplex grid.
    2. Plots the computational grid.
    3. Extracts the displacement jump along the fracture.
    4. Plots the displacement jump, compared with the analytical solution.
    5. Plots the x- and y-displacement fields.
    6. Saves all figures as compressed TIFF files.
"""
from __future__ import annotations

import porepy as pp
import numpy as np

from fems import FemWestergaardOpenMode2d

import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import os
save_dir = "figures/models/qfem/sol/wo/"
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
    "sigma_inf": 1e-4,    # far_field normal stress
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.05},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 45),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.0,
}

use_qpe = False
fem_model = FemWestergaardOpenMode2d(params, use_qpe)
fem_model.model_setup_and_sol()

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

# Plot the mesh
fig1, ax1 = plt.subplots(figsize=(3.05, 3.0))   # inches
ax1.triplot(
    fem_model.nodes_mdg[:,0], 
    fem_model.nodes_mdg[:,1], 
    fem_model.elements_mdg, 
    linewidth=0.3
)
ax1.xaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter(formatter)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_title("FEM mesh with the central fracture (T3)")
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
# ax1.set_xticks(np.arange(0.20, 0.8, 0.1))
# ax1.set_yticks(np.arange(1.0e-5, 5.01e-5, 1.0e-5))
# ax1.legend()
# ax1.grid(True, linestyle="--", alpha=0.5)
fig1.tight_layout()
fig1.savefig(
    os.path.join(save_dir, "QFEM_grids_wo_45d.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
# plt.show()
plt.close(fig1)

# Only for the horizontal fracture
x_coords, u_jump_normal, u_jump_tangential = fem_model.extract_displacement_jump()

# print("x_coords: ", x_coords)
# print("u_jump_normal: ", u_jump_normal.shape)
# print("u_jump_tangential: ", u_jump_tangential.shape)

# Get analytical solutions
eta_exact = np.abs(x_coords - fem_model.a)
u_open_analytical = fem_model.analytical_displacements(eta_exact)
u_open_numerical = u_jump_normal
u_slip_numerical = u_jump_tangential

fig2, ax2 = plt.subplots(figsize=(3.5, 3.0))   # inches
ax2.plot(
    x_coords,
    u_open_analytical,
    '-',
    color='tab:blue',
    linewidth=2, 
    label="Analytical"
)
ax2.plot(
    x_coords,
    u_open_numerical,
    'o',
    color='tab:orange',
    markersize=4, 
    label="Numerical"
)
ax2.xaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)
ax2.set_xlabel("Distance from fracture tip 1")
ax2.set_ylabel("Normal displacement jump")
ax2.set_title("Normal displacement jump")
ax2.set_xlim(0.00, 0.60)
ax2.set_ylim(1.0e-5, 5.0e-5)
ax2.set_xticks(np.arange(0.0, 0.601, 0.1))
ax2.set_yticks(np.arange(1.0e-5, 5.01e-5, 1.0e-5))
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig(
    os.path.join(save_dir, "QFEM_un_wo_45d.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
# plt.show()
plt.close(fig2)

ux = fem_model.U[0::2]
uy = fem_model.U[1::2]

x = fem_model.nodes_p6[:, 0]
y = fem_model.nodes_p6[:, 1]

Ufull = fem_model.U.ravel()   # (2N,)
ux = Ufull[0::2]
uy = Ufull[1::2]

uy = uy.ravel()   
ux = ux.ravel()

# Plot
triangles = []
for tri6 in fem_model.elements_t6:
    v0, m01, v1, m12, v2, m02 = tri6
    triangles.append([v0, m01, m02])
    triangles.append([m01, v1, m12])
    triangles.append([m02, m12, v2])
    triangles.append([m01, m12, m02])

triangles = np.array(triangles)

fig3, ax3 = plt.subplots(figsize=(3.5, 3.0))   # inches
# solution
tri_fine = mtri.Triangulation(fem_model.nodes_p6[:,0], fem_model.nodes_p6[:,1], triangles)
sol_fine = ax3.tripcolor(tri_fine, ux, shading="gouraud", cmap="jet")
fig3.colorbar(sol_fine, label="ux", shrink=0.8, aspect=20, pad=0.08)
# grids
tri_coarse = mtri.Triangulation(fem_model.nodes_mdg[:,0], fem_model.nodes_mdg[:,1], fem_model.elements_mdg)  
ax3.triplot(tri_coarse, color="k", linewidth=0.3)
ax3.set_aspect("equal")
ax3.xaxis.set_major_formatter(formatter)
ax3.yaxis.set_major_formatter(formatter)
ax3.set_xlabel("$x$")
ax3.set_ylabel("$y$")
ax3.set_title("FEM displacement solution ux")
ax3.set_xlim(0.0, 1.0)
ax3.set_ylim(0.0, 1.0)
fig3.tight_layout()
fig3.savefig(
    os.path.join(save_dir, "QFEM_ux_map_wo_45d.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
# plt.show()
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(3.5, 3.0))   # inches
# solution
tri_fine = mtri.Triangulation(fem_model.nodes_p6[:,0], fem_model.nodes_p6[:,1], triangles)
sol_fine = ax4.tripcolor(tri_fine, uy, shading="gouraud", cmap="jet")
fig4.colorbar(sol_fine, label="uy", shrink=0.8, aspect=20, pad=0.08)
# grids
tri_coarse = mtri.Triangulation(fem_model.nodes_mdg[:,0], fem_model.nodes_mdg[:,1], fem_model.elements_mdg)  
ax4.triplot(tri_coarse, color="k", linewidth=0.3)
ax4.set_aspect("equal")
ax4.xaxis.set_major_formatter(formatter)
ax4.yaxis.set_major_formatter(formatter)
ax4.set_xlabel("$x$")
ax4.set_ylabel("$y$")
ax4.set_title("FEM displacement solution uy")
ax4.set_xlim(0.0, 1.0)
ax4.set_ylim(0.0, 1.0)
fig4.tight_layout()
fig4.savefig(
    os.path.join(save_dir, "QFEM_uy_map_wo_45d.tiff"), 
    format="tiff", 
    dpi=300, 
    bbox_inches="tight", 
    pil_kwargs={"compression": "tiff_lzw"}
)
# plt.show()
plt.close(fig4)
