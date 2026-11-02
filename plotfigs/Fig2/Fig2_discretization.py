"""
Figure 2: Discretization illustrations of the 2D fractured domain
by three numerical methods: (a) QFEM, (b) QFEM-QPE, and (c) MPSA.
Black dots, purple dots, and green dots denote locations of displacement
unknowns, fracture tips, and quarter-point nodes, respectively. Fracture
tips coincide with displacement unknowns in both QFEM and QFEM-QPE,
whereas they are not represented by displacement unknowns in MPSA.
"""

from __future__ import annotations

import porepy as pp
import numpy as np

from sneddon_opening_fem import FemSneddonModel2d

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import os

save_dir = "figures/publication/Fig2/"
os.makedirs(save_dir, exist_ok=True)

a = 0.30
poi = 0.25
shear_modulus = 1.0
lam = 2 * shear_modulus * poi / (1 - 2 * poi)

params = {
    "prepare_simulation": True,
    "material_constants": {
        "solid": pp.SolidConstants(
            shear_modulus=shear_modulus,
            lame_lambda=lam,
        )
    },
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

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,
    "lines.markersize": 2.0,
    "mathtext.fontset": "dejavusans",
})

formatter = ScalarFormatter(useMathText=True)

def clean_axis(ax):
    """Remove ticks, labels, and spines."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal")


# Plot three subfigures in the whole figure.
fig1, axes = plt.subplots(1, 3, figsize=(7.0, 2.40))  # inches

# (a) QFEM.
ax1 = axes[0]

use_qpe = False
fem_model = FemSneddonModel2d(params, use_qpe)
fem_model.model_setup_and_sol()

ax1.triplot(
    fem_model.nodes_mdg[:, 0],
    fem_model.nodes_mdg[:, 1],
    fem_model.elements_mdg,
    linewidth=0.5,
    color="black",
    zorder=1,
)

theta_rad = params["theta_rad"]
frac_gap = 0.02

t_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)], dtype=float)
t_vec /= np.linalg.norm(t_vec)

n_vec = np.array([-np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
n_vec /= np.linalg.norm(n_vec)

orig_nodes = fem_model.grid2d.nodes[[0, 1], :].T

face_ind = np.reshape(
    fem_model.grid2d.face_nodes.indices,
    (fem_model.grid2d.dim, -1),
    order="F",
).T

frac_faces = np.where(fem_model.grid2d.tags["fracture_faces"])[0]
frac_node_ids = np.unique(face_ind[frac_faces].ravel())

s = orig_nodes[frac_node_ids] @ t_vec
order = np.argsort(s)
frac_node_ids = frac_node_ids[order]

centerline = orig_nodes[frac_node_ids].copy()

s_tip = centerline @ t_vec
tip_mask = np.zeros(len(centerline), dtype=bool)
tip_mask[np.argmin(s_tip)] = True
tip_mask[np.argmax(s_tip)] = True

tips = centerline[tip_mask]

ax1.scatter(
    tips[:, 0],
    tips[:, 1],
    color="tab:purple",
    s=14,
    zorder=20,
)

ax1.scatter(
    fem_model.nodes_p6[:, 0],
    fem_model.nodes_p6[:, 1],
    color="black",
    s=2,
    zorder=15,
)

clean_axis(ax1)

ax1.text(
    -0.05,
    0.98,
    "(a)",
    transform=ax1.transAxes,
    fontsize=9,
    ha="left",
    va="top",
)

# (b) QFEM-QPE.
ax1 = axes[1]

use_qpe = True
fem_model = FemSneddonModel2d(params, use_qpe)
fem_model.model_setup_and_sol()

ax1.triplot(
    fem_model.nodes_mdg[:, 0],
    fem_model.nodes_mdg[:, 1],
    fem_model.elements_mdg,
    linewidth=0.5,
    color="black",
    zorder=1,
)

theta_rad = params["theta_rad"]
frac_gap = 0.02

t_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)], dtype=float)
t_vec /= np.linalg.norm(t_vec)

n_vec = np.array([-np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
n_vec /= np.linalg.norm(n_vec)

orig_nodes = fem_model.grid2d.nodes[[0, 1], :].T

face_ind = np.reshape(
    fem_model.grid2d.face_nodes.indices,
    (fem_model.grid2d.dim, -1),
    order="F",
).T

frac_faces = np.where(fem_model.grid2d.tags["fracture_faces"])[0]
frac_node_ids = np.unique(face_ind[frac_faces].ravel())

s = orig_nodes[frac_node_ids] @ t_vec
order = np.argsort(s)
frac_node_ids = frac_node_ids[order]

centerline = orig_nodes[frac_node_ids].copy()

s_tip = centerline @ t_vec
tip_mask = np.zeros(len(centerline), dtype=bool)
tip_mask[np.argmin(s_tip)] = True
tip_mask[np.argmax(s_tip)] = True

tips = centerline[tip_mask]

ax1.scatter(
    tips[:, 0],
    tips[:, 1],
    color="tab:purple",
    s=14,
    zorder=20,
)

ax1.scatter(
    fem_model.nodes_p6[:, 0],
    fem_model.nodes_p6[:, 1],
    color="black",
    s=2,
    zorder=15,
)

# Highlight true quarter-point nodes.
qpe_node_ids = []
corner_locs = [0, 2, 4]

for qpe_group in fem_model.elements_quarter:
    for elem_id in qpe_group:
        tri = fem_model.elements_t6[elem_id]

        corner_ids = tri[corner_locs]
        corner_pts = fem_model.nodes_p6[corner_ids]

        dist_to_tips = np.linalg.norm(
            corner_pts[:, None, :] - fem_model.tip_face_centers[None, :, :],
            axis=2,
        )

        local_tip_corner_pos = np.unravel_index(
            np.argmin(dist_to_tips),
            dist_to_tips.shape,
        )[0]

        tip_local = corner_locs[local_tip_corner_pos]

        if tip_local == 0:
            qpe_node_ids.extend([tri[1], tri[5]])
        elif tip_local == 2:
            qpe_node_ids.extend([tri[1], tri[3]])
        elif tip_local == 4:
            qpe_node_ids.extend([tri[3], tri[5]])

qpe_node_ids = np.unique(qpe_node_ids)
qpe_nodes = fem_model.nodes_p6[qpe_node_ids]

ax1.scatter(
    qpe_nodes[:, 0],
    qpe_nodes[:, 1],
    color="tab:green",
    s=3,
    zorder=30,
)

clean_axis(ax1)

ax1.text(
    -0.05,
    0.98,
    "(b)",
    transform=ax1.transAxes,
    fontsize=9,
    ha="left",
    va="top",
)


# (c) MPSA.
ax1 = axes[2]

use_qpe = False
fem_model = FemSneddonModel2d(params, use_qpe)
fem_model.model_setup_and_sol()

ax1.triplot(
    fem_model.nodes_mdg[:, 0],
    fem_model.nodes_mdg[:, 1],
    fem_model.elements_mdg,
    linewidth=0.5,
    color="black",
    zorder=1,
)

theta_rad = params["theta_rad"]

t_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)], dtype=float)
t_vec /= np.linalg.norm(t_vec)

n_vec = np.array([-np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
n_vec /= np.linalg.norm(n_vec)

nodes = fem_model.nodes_mdg

upper_edges = fem_model.upper_frac_edges.copy()
lower_edges = fem_model.lower_frac_edges.copy()

upper_edge_centers = 0.5 * (
    nodes[upper_edges[:, 0]] + nodes[upper_edges[:, 1]]
)
lower_edge_centers = 0.5 * (
    nodes[lower_edges[:, 0]] + nodes[lower_edges[:, 1]]
)

upper_order = np.argsort(upper_edge_centers @ t_vec)
lower_order = np.argsort(lower_edge_centers @ t_vec)

upper_edges = upper_edges[upper_order]
lower_edges = lower_edges[lower_order]

upper_edge_centers = upper_edge_centers[upper_order]
lower_edge_centers = lower_edge_centers[lower_order]


def ordered_nodes_from_edges(edges: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return ordered node coordinates from fracture boundary edges."""
    edge_nodes = np.unique(edges.ravel())
    pts = nodes[edge_nodes]
    return pts[np.argsort(pts @ t_vec)]


upper_interface = ordered_nodes_from_edges(upper_edges, upper_edge_centers)
lower_interface = ordered_nodes_from_edges(lower_edges, lower_edge_centers)

tips = fem_model.tip_face_centers

ax1.scatter(
    tips[:, 0],
    tips[:, 1],
    color="tab:purple",
    s=14,
    zorder=20,
)

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

clean_axis(ax1)

ax1.text(
    -0.05,
    0.98,
    "(c)",
    transform=ax1.transAxes,
    fontsize=9,
    ha="left",
    va="top",
)

fig1.tight_layout(w_pad=0.1)

# Save.
fig1.savefig(
    os.path.join(save_dir, "Fig2_discretization.pdf"),
    format="pdf",
)

fig1.savefig(
    os.path.join(save_dir, "Fig2_discretization.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

fig1.savefig(
    os.path.join(save_dir, "Fig2_discretization.png"),
    format="png",
    dpi=600,
)

plt.close(fig1)
