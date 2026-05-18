"""
Solve the model for Sneddon's problem by a quadratic finite element method (FEM)
with quarter-point elements (QPEs).
Quarter-point elements (QPEs) can be activated based on the `use_qpe` flag.

"""
from __future__ import annotations

import porepy as pp
import numpy as np

from sneddon_opening_fem import FemSneddonModel2d

import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import os
save_dir = "figures/illustration/"
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
    "p0": 1e-4,
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.20},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 45),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.0,
}

use_qpe = True
# use_qpe = False
fem_model = FemSneddonModel2d(params, use_qpe)
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
fig1, ax1 = plt.subplots(figsize=(3.0, 3.0))   # inches
ax1.triplot(
    fem_model.nodes_mdg[:,0], 
    fem_model.nodes_mdg[:,1], 
    fem_model.elements_mdg, 
    linewidth=0.5,
    color="black",
    zorder=1,
)
# Explicitly plot detached fracture interface lines
# Explicitly plot split matrix internal fracture boundaries
theta_rad = params["theta_rad"]

t_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)], dtype=float)
t_vec /= np.linalg.norm(t_vec)

n_vec = np.array([-np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
n_vec /= np.linalg.norm(n_vec)

# -------------------------------------------------------------------------
# Use the already split FEM/MPSA-style matrix mesh.
# These nodes are consistent with ax1.triplot(fem_model.nodes_mdg, ...).
# -------------------------------------------------------------------------
nodes = fem_model.nodes_mdg

upper_edges = fem_model.upper_frac_edges.copy()
lower_edges = fem_model.lower_frac_edges.copy()

# Edge centers on the two expanded matrix internal boundaries.
upper_edge_centers = 0.5 * (
    nodes[upper_edges[:, 0]] + nodes[upper_edges[:, 1]]
)
lower_edge_centers = 0.5 * (
    nodes[lower_edges[:, 0]] + nodes[lower_edges[:, 1]]
)

# Sort fracture boundary edges along the fracture direction.
upper_order = np.argsort(upper_edge_centers @ t_vec)
lower_order = np.argsort(lower_edge_centers @ t_vec)

upper_edges = upper_edges[upper_order]
lower_edges = lower_edges[lower_order]

upper_edge_centers = upper_edge_centers[upper_order]
lower_edge_centers = lower_edge_centers[lower_order]

# -------------------------------------------------------------------------
# Construct ordered upper/lower internal boundary nodes from ordered edges.
# -------------------------------------------------------------------------
def ordered_nodes_from_edges(edges: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return ordered node coordinates from fracture boundary edges."""
    edge_nodes = np.unique(edges.ravel())
    pts = nodes[edge_nodes]
    return pts[np.argsort(pts @ t_vec)]


upper_interface = ordered_nodes_from_edges(upper_edges, upper_edge_centers)
lower_interface = ordered_nodes_from_edges(lower_edges, lower_edge_centers)

# Tips are taken from the original fracture tips.
tips = fem_model.tip_face_centers

# -------------------------------------------------------------------------
# Plot the two split matrix internal boundaries.
# -------------------------------------------------------------------------
# ax1.plot(
#     upper_interface[:, 0],
#     upper_interface[:, 1],
#     color="tab:orange",
#     linewidth=0.6,
#     zorder=10,
# )

# ax1.plot(
#     lower_interface[:, 0],
#     lower_interface[:, 1],
#     color="tab:orange",
#     linewidth=0.6,
#     zorder=10,
# )

ax1.scatter(
    tips[:, 0],
    tips[:, 1],
    color="tab:purple",
    s=14,
    zorder=20,
)

# -------------------------------------------------------------------------
# Plot matrix cell centers.
# -------------------------------------------------------------------------
matrix_cell_centers = np.mean(
    fem_model.nodes_mdg[fem_model.elements_mdg],
    axis=1,
)

ax1.scatter(
    matrix_cell_centers[:, 0],
    matrix_cell_centers[:, 1],
    color="black",
    s=2,
    zorder=15,
)

# -------------------------------------------------------------------------
# Plot fracture/interface cell centers on the split matrix internal boundary.
# These are transferred from the 1D fracture/interface to the matrix boundary.
# -------------------------------------------------------------------------
upper_cell_centers = upper_edge_centers
lower_cell_centers = lower_edge_centers

ax1.scatter(
    upper_cell_centers[:, 0],
    upper_cell_centers[:, 1],
    color="black",
    s=2,
    zorder=16,
)

ax1.scatter(
    lower_cell_centers[:, 0],
    lower_cell_centers[:, 1],
    color="black",
    s=2,
    zorder=16,
)

# -------------------------------------------------------------------------
# Plot short orange separators on the split matrix internal boundaries.
# -------------------------------------------------------------------------
# sep_len = 0.015
# 
# for p in upper_interface:
#     if np.min(np.linalg.norm(tips - p[None, :], axis=1)) < 1e-8:
#         continue

#     ax1.plot(
#         [p[0] - n_vec[0] * sep_len / 2, p[0] + n_vec[0] * sep_len / 2],
#         [p[1] - n_vec[1] * sep_len / 2, p[1] + n_vec[1] * sep_len / 2],
#         color="tab:orange",
#         linewidth=0.6,
#         zorder=18,
#     )

# for p in lower_interface:
#     if np.min(np.linalg.norm(tips - p[None, :], axis=1)) < 1e-8:
#         continue

#     ax1.plot(
#         [p[0] - n_vec[0] * sep_len / 2, p[0] + n_vec[0] * sep_len / 2],
#         [p[1] - n_vec[1] * sep_len / 2, p[1] + n_vec[1] * sep_len / 2],
#         color="tab:orange",
#         linewidth=0.6,
#         zorder=18,
#     )
# -------------------------------------------------------------------------
# Plot the merged 1D fracture/interface grid in the crack opening.
# This is only for visualization. Its DOFs/cell centers are shown in the
# middle between the two expanded matrix internal boundaries.
# -------------------------------------------------------------------------

n_frac_cells = min(len(upper_edge_centers), len(lower_edge_centers))

fracture_cell_centers = 0.5 * (
    upper_edge_centers[:n_frac_cells] + lower_edge_centers[:n_frac_cells]
)

# Construct fracture/interface nodes as the midpoint of upper/lower
# split matrix boundary nodes.
n_frac_nodes = min(len(upper_interface), len(lower_interface))

fracture_line_nodes = 0.5 * (
    upper_interface[:n_frac_nodes] + lower_interface[:n_frac_nodes]
)

# Sort again for safety.
fracture_line_nodes = fracture_line_nodes[
    np.argsort(fracture_line_nodes @ t_vec)
]

fracture_cell_centers = fracture_cell_centers[
    np.argsort(fracture_cell_centers @ t_vec)
]

# Draw the central fracture/interface line.
ax1.plot(
    fracture_line_nodes[:, 0],
    fracture_line_nodes[:, 1],
    color="tab:orange",
    linewidth=0.8,
    zorder=17,
)

# Draw cell centers on the central fracture/interface line.
ax1.scatter(
    fracture_cell_centers[:, 0],
    fracture_cell_centers[:, 1],
    color="blue",
    s=2,
    marker="D",
    zorder=19,
)

# Draw short orange separators on the central fracture/interface line.
sep_len = 0.015

for p in fracture_line_nodes[1:-1]:
    ax1.plot(
        [p[0] - n_vec[0] * sep_len / 2, p[0] + n_vec[0] * sep_len / 2],
        [p[1] - n_vec[1] * sep_len / 2, p[1] + n_vec[1] * sep_len / 2],
        color="tab:orange",
        linewidth=0.7,
        zorder=20,
    )

# Remove ticks / labels, keep only square frame
ax1.set_xlabel("")
ax1.set_ylabel("")

ax1.set_xticks([])
ax1.set_yticks([])

ax1.tick_params(
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)

for spine in ax1.spines.values():
    spine.set_visible(False)

fig1.tight_layout()
# fig1.savefig(
#     os.path.join(save_dir, "FEM_mesh_045d.tiff"), 
#     format="tiff", 
#     dpi=300, 
#     bbox_inches="tight", 
#     pil_kwargs={"compression": "tiff_lzw"}
# )
fig1.savefig(
    os.path.join(save_dir, "MPSA2.pdf"), 
    format="pdf", 
    bbox_inches="tight", 
)
plt.close(fig1)

