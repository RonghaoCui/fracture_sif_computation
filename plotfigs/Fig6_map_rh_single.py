from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D


save_fig = "figures/publication/Fig6/"
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
    """Read averaged-map data."""
    return pd.read_csv(csv_file)


def prepare_mean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean error for each h-r pair."""
    return (
        df.groupby(["norm_h", "r"], as_index=False)
        .agg(mean_error=("error", "mean"))
    )


def plot_case(fig, ax, df: pd.DataFrame, number: str):
    """Plot a scatter subplot."""

    df_mean = prepare_mean_data(df)

    vmin = df_mean["mean_error"].min()
    vmax = df_mean["mean_error"].max()

    sc = ax.scatter(
        df_mean["r"],
        df_mean["norm_h"],
        c=df_mean["mean_error"],
        s=18,
        cmap="plasma",
        linewidths=0,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    ax.set_yscale("log")

    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$h/a$")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(
        0.9 * df_mean["norm_h"].min(),
        1.1 * df_mean["norm_h"].max(),
    )

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )

    # ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.text(
        -0.15,
        0.98,
        number,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        location="bottom",
        shrink=0.80,
        pad=0.25,
        fraction=0.05,
    )
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)

    return sc


# Read data
case_files = [
    {
        "csv": "data/sifconv/cache_modes/qfem/so/map/map_so.csv",
        "title": "QFEM, Sneddon $K_I$",
    },
    {
        "csv": "data/sifconv/cache_modes/qfem_qpe/so/map/map_so.csv",
        "title": "QFEM-QPE, Sneddon $K_I$",
    },
    {
        "csv": "data/sifconv/cache_modes/mpsa/so/map/map_so.csv",
        "title": "MPSA, Sneddon $K_I$",
    },
    {
        "csv": "data/sifconv/cache_modes/qfem/wo/map/map_wo.csv",
        "title": "QFEM, Westergaard $K_{I}$",
    },
    {
        "csv": "data/sifconv/cache_modes/qfem_qpe/wo/map/map_wo.csv",
        "title": "QFEM-QPE, Westergaard $K_{I}$",
    },
    {
        "csv": "data/sifconv/cache_modes/mpsa/wo/map/map_wo.csv",
        "title": "MPSA, Westergaard $K_{I}$",
    },
    {
        "csv": "data/sifconv/cache_modes/qfem/ws/map/map_ws.csv",
        "title": "QFEM, Westergaard $K_{II}$",
    },
    {
        "csv": "data/sifconv/cache_modes/qfem_qpe/ws/map/map_ws.csv",
        "title": "QFEM-QPE, Westergaard $K_{II}$",
    },
    {
        "csv": "data/sifconv/cache_modes/mpsa/ws/map/map_ws.csv",
        "title": "MPSA, Westergaard $K_{II}$",
    },
]


# Read all data
all_data = [read_case(item["csv"]) for item in case_files]

# Use a shared color scale for all six subplots
all_mean_errors = []

for df in all_data:
    df_mean = prepare_mean_data(df)
    all_mean_errors.append(df_mean["mean_error"].values)

all_mean_errors = np.concatenate(all_mean_errors)

vmin = all_mean_errors.min()
vmax = all_mean_errors.max()
vmin = 2e-3
vmax = 1e-1

# Plot three horizontal subfigures
fig, axes = plt.subplots(
    3,
    3,
    figsize=(6.8, 7.0),
    sharey=False,
)

axes_flat = axes.ravel()

sc = None
numbers = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
for ax, df, item, number in zip(axes_flat, all_data, case_files, numbers):
    sc = plot_case(
        fig=fig,
        ax=ax,
        df=df,
        number=number,
    )

# Shared colorbar
# cbar = fig.colorbar(
#     sc,
#     ax=axes_flat,
#     location="bottom",
#     shrink=0.60,
#     pad=0.08,
#     fraction=0.05,
# )

# cbar.set_label("Mean of relative error")

# Adjust layout to make room for the legend
# plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.05,
    top=0.95,
    wspace=0.40,
    hspace=0.15,
)

plt.savefig(
    os.path.join(save_fig, "Fig6_Map_r_h_single.pdf"),
    format="pdf",
)

plt.savefig(
    os.path.join(save_fig, "Fig6_Map_r_h_single.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.close(fig)
