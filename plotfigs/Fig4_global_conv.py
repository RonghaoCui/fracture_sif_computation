from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D


save_fig = "figures/publication/Fig4_2/"
os.makedirs(save_fig, exist_ok=True)


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


def read_case(base_path: str, method_name: str):
    """Read four self-convergence cases."""
    data = {}

    for case in ["so", "wo", "ws", "wm"]:
        file_name = f"globalconv_{method_name}_{case}.csv"
        csv_file = os.path.join(base_path, case, file_name)

        df = pd.read_csv(csv_file)

        data[case] = {
            "sqrt_dofs": np.sqrt(df["matrix_dof"].values),
            "matrix_error": df["matrix_error_quad"].values,
            "fracture_error": df["fracture_error_quad"].values,
        }

    return data


def add_reference_triangle(ax, x0, x1, y0, slope=1.5):
    """Add reference slope triangle."""
    y1 = y0 * (x0 / x1) ** slope

    triangle_x = [x0, x0, x1]
    triangle_y = [y0, y1, y1]

    ax.fill(
        triangle_x,
        triangle_y,
        color="gray",
        alpha=0.5,
        zorder=1,
    )

    triangle_style = dict(
        color="black",
        linewidth=0.6,
        zorder=2,
    )

    ax.plot([x0, x1], [y1, y1], **triangle_style)
    ax.plot([x0, x0], [y0, y1], **triangle_style)
    ax.plot([x0, x1], [y0, y1], **triangle_style)

    ax.text(
        x0 / 1.08,
        np.sqrt(y0 * y1),
        f"{slope:.1f}",
        ha="right",
        va="center",
    )

    ax.text(
        np.sqrt(x0 * x1),
        y1 / 1.30,
        "1.0",
        ha="center",
        va="top",
    )


def plot_method(ax, data, title):
    """Plot one method in one subplot."""

    plot_info = {
        "so": ("tab:blue", "Sneddon $K_I$"),
        "wo": ("tab:orange", "Westergaard $K_I$"),
        "ws": ("tab:pink", "Westergaard $K_{II}$"),
        "wm": ("tab:green", "Westergaard mixed"),
    }

    for case, (color, label) in plot_info.items():
        x = data[case]["sqrt_dofs"]

        ax.loglog(
            x,
            data[case]["matrix_error"],
            "-",
            color=color,
            label=label,
        )

        ax.loglog(
            x,
            data[case]["fracture_error"],
            "--",
            color=color,
            label="_nolegend_",
        )

    ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")

    ax.set_ylabel("Relative error")

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(
        -0.33,
        1.05,
        title,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )


# Read data
qfem_data = read_case(
    base_path="data/global_conv/qfem",
    method_name="qfem",
)

qfem_qpe_data = read_case(
    base_path="data/global_conv/qfem_qpe",
    method_name="qfem_qpe",
)

mpsa_data = read_case(
    base_path="data/global_conv/mpsa",
    method_name="mpsa",
)


# Plot three horizontal subfigures
fig, axes = plt.subplots(
    1,
    3,
    figsize=(6.8, 2.4),
    sharey=False,
)

# QFEM
plot_method(axes[0], qfem_data, "(a)")
# QFEM-QPE
plot_method(axes[1], qfem_qpe_data, "(b)")
# MPSA
plot_method(axes[2], mpsa_data, "(c)")

# Axis limits
axes[0].set_xlim(10, 1000)
axes[0].set_ylim(1e-5, 1.0)

axes[1].set_xlim(10, 1000)
axes[1].set_ylim(1e-5, 1.0)

axes[2].set_xlim(10, 1000)
axes[2].set_ylim(1e-5, 1.0)


# Reference triangles
add_reference_triangle(axes[0], x0=25, x1=45, y0=1e-4)
add_reference_triangle(axes[1], x0=25, x1=45, y0=1e-4)
add_reference_triangle(axes[2], x0=25, x1=45, y0=1e-4)


# Color legend
handles = [

    # col1
    Line2D([], [], color='none', label='Displacement:'),
    Line2D([], [], color='none', label='Displacement jump:'),

    # col2
    Line2D([], [], color="tab:blue", ls="-",  lw=1.0,
           label=r"Sneddon $K_I$"),
    Line2D([], [], color="tab:blue", ls="--", lw=1.0,
           label=r"Sneddon $K_I$"),

    # col3
    Line2D([], [], color="tab:orange", ls="-",  lw=1.0,
           label=r"Westergaard $K_I$"),
    Line2D([], [], color="tab:orange", ls="--", lw=1.0,
           label=r"Westergaard $K_I$"),

    # col4
    Line2D([], [], color="tab:pink", ls="-",  lw=1.0,
           label=r"Westergaard $K_{II}$"),
    Line2D([], [], color="tab:pink", ls="--", lw=1.0,
           label=r"Westergaard $K_{II}$"),

    # col5
    Line2D([], [], color="tab:green", ls="-",  lw=1.0,
           label="Westergaard Mixed"),
    Line2D([], [], color="tab:green", ls="--", lw=1.0,
           label="Westergaard Mixed"),
]

legend = fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=5,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
)

# Adjust layout to make room for the legend
# plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.35,
    top=0.95,
    wspace=0.40,
)

plt.savefig(
    os.path.join(save_fig, "Fig4_Global_conv.pdf"),
    format="pdf",
)

plt.savefig(
    os.path.join(save_fig, "Fig4_Global_conv.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.close(fig)
