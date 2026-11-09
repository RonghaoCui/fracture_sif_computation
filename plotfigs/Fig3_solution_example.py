"""Figure 3: Solution of displacement components, u_x (a) and u_y (b)
in Westergaard's problems of mixed modes. Corresponding normal and
tangential displacement jump profiles are shown in (c) and (d), respectively.
MPSA discretization is adopted as an example.
"""
from __future__ import annotations

import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.tri as mtri

import numpy as np
import porepy as pp

from westergaard2d import WestergaardMixedMode2d

save_dir = "figures/publication/Fig3/"
os.makedirs(save_dir, exist_ok=True)

a = 0.3
poi = 0.25
shear_modulus = 1.0
lam = 2 * shear_modulus * poi / (1 - 2 * poi)

params = {
    "prepare_simulation": True,
    "material_constants": {
        "solid": pp.SolidConstants(
            shear_modulus=shear_modulus, lame_lambda=lam
        )
    },
    "a": a,
    "domain_size": 1.0,
    "meshing_arguments": {"cell_size": 0.025},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 45),
    "num_bem_segments": 1000,
    "poi": poi,
    "error_exclusion_zone_fracture_tips": 0.0,
    "sigma_inf": 1e-4,
    "tau_inf": 1e-4,
    "mu": 0.0,
    "p0": 0.0,
}

model = WestergaardMixedMode2d(params=params)
pp.run_time_dependent_model(model)

matrix_grid = model.mdg.subdomains(dim=2)[0]
u = model.displacement([matrix_grid]).value(model.equation_system)
u_matrix = u.reshape((2, matrix_grid.num_cells), order="F")


def cell_polygons(grid: pp.Grid) -> list[np.ndarray]:
    """Return the physical vertices of every matrix cell."""
    cell_nodes = grid.cell_nodes().tocsc()
    polygons = []
    for cell in range(grid.num_cells):
        nodes = cell_nodes.indices[
            cell_nodes.indptr[cell] : cell_nodes.indptr[cell + 1]
        ]
        vertices = grid.nodes[:2, nodes].T
        centroid = np.mean(vertices, axis=0)
        angle = np.arctan2(
            vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0]
        )
        polygons.append(vertices[np.argsort(angle)])
    return polygons


polygons = cell_polygons(matrix_grid)

cell_nodes = matrix_grid.cell_nodes().tocsc()

triangles = []
for cell in range(matrix_grid.num_cells):
    nodes = cell_nodes.indices[
        cell_nodes.indptr[cell] : cell_nodes.indptr[cell + 1]
    ]

    triangles.append(nodes)

triangles = np.asarray(triangles)

triangulation = mtri.Triangulation(
    matrix_grid.nodes[0, :],
    matrix_grid.nodes[1, :],
    triangles,
)

# Fracture displacement jump. This includes the
# orientation correction for PorePy's local tangential basis.
fracture_grid = model.mdg.subdomains(dim=1)[0]
theta = params["theta_rad"]
t_vec_frac = np.array([np.sin(theta), np.cos(theta)])
coords = fracture_grid.cell_centers[:2, :]
tip_faces = fracture_grid.tags["tip_faces"].nonzero()[0]
tip_centers = fracture_grid.face_centers[:2, tip_faces]
tip0 = tip_centers[:, np.argmin(t_vec_frac @ tip_centers)]
x_plot = t_vec_frac @ (coords - tip0[:, None])

u_jump = model.displacement_jump([fracture_grid])
normal_op = model.normal_component([fracture_grid])
tangent_op = model.tangential_component([fracture_grid])
u_open_numerical = (normal_op @ u_jump).value(model.equation_system)
sign_t = 1.0 if np.sin(theta) >= 0 else -1.0
u_slip_numerical = sign_t * (tangent_op @ u_jump).value(model.equation_system)
u_open_analytical = model.exact_sol.exact_sol_normal_fracture(model.mdg)
u_slip_analytical = model.exact_sol.exact_sol_tangential_fracture(model.mdg)

order = np.argsort(x_plot)
x_plot = x_plot[order]
u_open_numerical = u_open_numerical[order]
u_slip_numerical = u_slip_numerical[order]
u_open_analytical = u_open_analytical[order]
u_slip_analytical = u_slip_analytical[order]

mpl.rcParams.update(
    {
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
    }
)

fig = plt.figure(
    figsize=(5.5, 4.0),
    constrained_layout=True,
)

gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1, 1, 0.05],
    height_ratios=[1, 1],
)

ax_ux = fig.add_subplot(gs[0, 0])
ax_uy = fig.add_subplot(gs[0, 1])

# Colorbar for displacement.
cax = fig.add_subplot(gs[0, 2])
pos = cax.get_position()
cax.set_position([
    pos.x0 - 0.05,
    pos.y0 + 0.20 * pos.height,
    pos.width,
    0.9 * pos.height,
])

ax_un = fig.add_subplot(gs[1, 0])
ax_ut = fig.add_subplot(gs[1, 1])

# Legend for displacement jump.
lax   = fig.add_subplot(gs[1, 2])

limit = max(
    np.max(np.abs(u_matrix[0])),
    np.max(np.abs(u_matrix[1]))
)

def plot_displacement_map(ax, values: np.ndarray, component: str):
    """Plot one displacement component over the matrix domain."""
    collection = ax.tripcolor(
        triangulation,
        facecolors=np.asarray(values),
        shading="flat",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        edgecolors="face",
        linewidth=0.001,
        antialiased=False,
    )

    frac_nodes = fracture_grid.nodes[:2, :]
    frac_order = np.argsort(t_vec_frac @ frac_nodes)

    ax.plot(
        frac_nodes[0, frac_order],
        frac_nodes[1, frac_order],
        color="k",
        linewidth=0.3,
        zorder=10,
    )

    ax.set_aspect("equal")
    ax.set_xlim(0.0, params["domain_size"])
    ax.set_ylim(0.0, params["domain_size"])
    ticks = np.linspace(0.0, params["domain_size"], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    return collection


collection_ux = plot_displacement_map(ax_ux, u_matrix[0], "x")
collection_uy = plot_displacement_map(ax_uy, u_matrix[1], "y")

# Use the same colorbar for both displacement components.
colorbar = fig.colorbar(collection_uy, cax=cax,)
colorbar.set_label(rf"Displacement")
colorbar.formatter = ScalarFormatter(useMathText=True)
colorbar.formatter.set_powerlimits((0, 0))
colorbar.update_ticks()
offset_text = colorbar.ax.yaxis.get_offset_text()
offset_text.set_x(4.0)
offset_text.set_y(1.01)


def plot_jump(ax, analytical, numerical, ylabel: str):
    ax.plot(x_plot, analytical, "-", color="tab:blue", linewidth=1.0,
            label="Analytical")
    ax.plot(x_plot, numerical, "o", color="tab:orange", markersize=2,
            label="Numerical")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Distance from the left tip")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 0.60)
    ax.set_ylim(1.0e-5, 5.0e-5)
    ax.set_xticks(np.arange(0.0, 0.61, 0.2))
    ax.set_yticks(np.arange(1.0e-5, 5.01e-5, 1.0e-5))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_box_aspect(0.85)


plot_jump(ax_un, u_open_analytical, u_open_numerical,
          "Normal\ndisplacement jump")
plot_jump(ax_ut, u_slip_analytical, u_slip_numerical,
          "Tangential\ndisplacement jump")

# Shared legend for (c) and (d)
handles, labels = (
    ax_un.get_legend_handles_labels()
)

lax.axis("off")
lax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
)

for ax, label in zip((ax_ux, ax_uy, ax_un, ax_ut), ("(a)", "(b)", "(c)", "(d)")):
    ax.text(
        -0.42, 1.05, 
        label, 
        transform=ax.transAxes, 
        ha="left", 
        va="top",
        fontsize=9, 
        clip_on=False,
    )

fig.savefig(
    os.path.join(save_dir, "Fig3_solution_example.pdf"), 
    format="pdf",
)

fig.savefig(
    os.path.join(save_dir, "Fig3_solution_example.tiff"), 
    format="tiff", 
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

fig.savefig(
    os.path.join(save_dir, "Fig3_solution_example.png"), 
    format="png", 
    dpi=600,
)

plt.close(fig)
