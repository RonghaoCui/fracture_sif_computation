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

use_qpe = False
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
theta_rad = params["theta_rad"]
frac_gap = 0.02

t_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)], dtype=float)
t_vec /= np.linalg.norm(t_vec)

n_vec = np.array([-np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
n_vec /= np.linalg.norm(n_vec)

# Original unsplit PorePy nodes
orig_nodes = fem_model.grid2d.nodes[[0, 1], :].T

face_ind = np.reshape(
    fem_model.grid2d.face_nodes.indices,
    (fem_model.grid2d.dim, -1),
    order="F",
).T

frac_faces = np.where(fem_model.grid2d.tags["fracture_faces"])[0]
frac_node_ids = np.unique(face_ind[frac_faces].ravel())

# Sort fracture nodes along fracture direction
s = orig_nodes[frac_node_ids] @ t_vec
order = np.argsort(s)
frac_node_ids = frac_node_ids[order]

centerline = orig_nodes[frac_node_ids].copy()

# Identify tip nodes
s_tip = centerline @ t_vec
tip_mask = np.zeros(len(centerline), dtype=bool)
tip_mask[np.argmin(s_tip)] = True
tip_mask[np.argmax(s_tip)] = True

upper_interface = centerline.copy()
lower_interface = centerline.copy()

upper_interface[~tip_mask] += n_vec * frac_gap / 2
lower_interface[~tip_mask] -= n_vec * frac_gap / 2

# ax1.plot(
#     upper_interface[:, 0],
#     upper_interface[:, 1],
#     color="orange",
#     linewidth=0.6,
#     zorder=10,
# )

# ax1.plot(
#     lower_interface[:, 0],
#     lower_interface[:, 1],
#     color="orange",
#     linewidth=0.6,
#     zorder=10,
# )

tips = centerline[tip_mask]

ax1.scatter(
    tips[:, 0],
    tips[:, 1],
    color="tab:purple",
    s=14,
    zorder=20,
)

# matrix_cell_centers = np.mean(
#     fem_model.nodes_mdg[fem_model.elements_mdg],
#     axis=1,
# )

# ax1.scatter(
#     matrix_cell_centers[:, 0],
#     matrix_cell_centers[:, 1],
#     color="black",
#     s=2,
#     zorder=15,
# )

ax1.scatter(
    fem_model.nodes_p6[:, 0],
    fem_model.nodes_p6[:, 1],
    color="black",
    s=2,
    zorder=15,
)

# upper_centers = 0.5 * (upper_interface[:-1] + upper_interface[1:])
# lower_centers = 0.5 * (lower_interface[:-1] + lower_interface[1:])

# upper_cell_centers = upper_centers[::2]
# lower_cell_centers = lower_centers[::2]

# ax1.scatter(
#     upper_cell_centers[:, 0],
#     upper_cell_centers[:, 1],
#     color="black",
#     s=3,
#     zorder=16,
# )

# ax1.scatter(
#     lower_cell_centers[:, 0],
#     lower_cell_centers[:, 1],
#     color="black",
#     s=3,
#     zorder=16,
# )

# # Plot fracture interface faces as short orange separators
# sep_len = 0.015  # adjust for visual effect
# for p in upper_interface[~tip_mask]:
#     ax1.plot(
#         [p[0] - n_vec[0] * sep_len / 2, p[0] + n_vec[0] * sep_len / 2],
#         [p[1] - n_vec[1] * sep_len / 2, p[1] + n_vec[1] * sep_len / 2],
#         color="orange",
#         linewidth=0.6,
#         zorder=18,
#     )

# for p in lower_interface[~tip_mask]:
#     ax1.plot(
#         [p[0] - n_vec[0] * sep_len / 2, p[0] + n_vec[0] * sep_len / 2],
#         [p[1] - n_vec[1] * sep_len / 2, p[1] + n_vec[1] * sep_len / 2],
#         color="orange",
#         linewidth=0.6,
#         zorder=18,
#     )

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
    os.path.join(save_dir, "QFEM.pdf"), 
    format="pdf", 
    bbox_inches="tight", 
)
plt.close(fig1)

