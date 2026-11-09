"""Figure 11: The convergence behavior of mode-I SIF estimates in
panels (a)-(c) and of mode-II SIF estimates in panels (d)-(e) across
different discretization methods using scaling-law sampling. Panels (a),
(b), and (d) are Sneddon's problem, Westergaard's mode-I problem,
and Westergaard's mode-II problem, respectively. Panels (c) and (e)
are both Westergaard's mixed-mode problem.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


save_fig = "figures/publication/Fig11/"
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


def read_case(csv_file: str) -> pd.DataFrame:
    """Read convergence data."""
    return pd.read_csv(csv_file)


def make_method_files(case_key: str) -> list[dict]:
    """Create file paths for QFEM, QFEM-QPE, and MPSA."""

    case_dir = {
        "so": "so",
        "wo": "wo",
        "wmo": "wm",
        "ws": "ws",
        "wms": "wm",
    }[case_key]

    return [
        {
            "csv": f"data/sifconv/cache_modes/qfem/{case_dir}/conv/conv_sif_{case_key}_qfem_scaling.csv",
            "method": "QFEM",
        },
        {
            "csv": f"data/sifconv/cache_modes/qfem_qpe/{case_dir}/conv/conv_sif_{case_key}_qfem_qpe_scaling.csv",
            "method": "QFEM-QPE",
        },
        {
            "csv": f"data/sifconv/cache_modes/mpsa/{case_dir}/conv/conv_sif_{case_key}_mpsa_scaling.csv",
            "method": "MPSA",
        },
    ]


def plot_case(ax, data_items: list[dict], number: str):
    """Plot one convergence subplot."""

    plot_info = {
        "QFEM": "tab:blue",
        "QFEM-QPE": "tab:orange",
        "MPSA": "tab:green",
    }

    for item in data_items:
        method = item["method"]
        df = read_case(item["csv"])
        color = plot_info[method]

        x = df["sqrt_dof"].to_numpy()
        y = df["error_ave"].to_numpy()
        y_std = df["std_error_ave"].to_numpy()

        y_lower = np.maximum(y - y_std, 1e-16)
        y_upper = y + y_std

        ax.loglog(
            x,
            y,
            "-",
            color=color,
            linewidth=1.0,
            label=method,
        )

        ax.fill_between(
            x,
            y_lower,
            y_upper,
            color=color,
            alpha=0.20,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
    ax.set_ylabel("Relative error")

    ax.set_xlim(10, 1000)
    ax.set_ylim(1e-5, 1.0)

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.text(
        -0.32,
        1.05,
        number,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )


panel_cases = [
    {"number": "(a)", "case_key": "so"},
    {"number": "(b)", "case_key": "wo"},
    {"number": "(c)", "case_key": "wmo"},
    {"number": "(d)", "case_key": "ws"},
    {"number": "(e)", "case_key": "wms"},
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(6.8, 4.6),
    sharey=False,
)

axes_flat = axes.ravel()

for ax, panel in zip(axes_flat[:5], panel_cases):
    plot_case(
        ax=ax,
        data_items=make_method_files(panel["case_key"]),
        number=panel["number"],
    )

# Legend panel.
ax_leg = axes_flat[5]
ax_leg.axis("off")

legend_handles = [
    Line2D([], [], color="tab:blue", linestyle="-", label="QFEM"),
    Patch(facecolor="tab:blue", alpha=0.20, edgecolor="none",
          label=r"QFEM: $\pm 1$ STD"),

    Line2D([], [], color="tab:orange", linestyle="-", label="QFEM-QPE"),
    Patch(facecolor="tab:orange", alpha=0.20, edgecolor="none",
          label=r"QFEM-QPE: $\pm 1$ STD"),

    Line2D([], [], color="tab:green", linestyle="-", label="MPSA"),
    Patch(facecolor="tab:green", alpha=0.20, edgecolor="none",
          label=r"MPSA: $\pm 1$ STD"),
]

ax_leg.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    handlelength=1.8,
    labelspacing=0.6,
)

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.10,
    top=0.95,
    wspace=0.38,
    hspace=0.36,
)

plt.savefig(
    os.path.join(save_fig, "Fig11_scaling_law_convergence.pdf"),
    format="pdf",
)

plt.savefig(
    os.path.join(save_fig, "Fig11_scaling_law_convergence.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.savefig(
    os.path.join(save_fig, "Fig11_scaling_law_convergence.png"),
    format="png",
    dpi=600,
)

plt.close(fig)
