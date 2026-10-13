from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D


save_fig = "figures/publication/Fig9/"
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


def read_case(mean_csv: str, theta_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read averaged-map data and angle-dependent data."""
    return pd.read_csv(mean_csv), pd.read_csv(theta_csv)


def prepare_mean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean error for each h-r pair."""
    return (
        df.groupby(["norm_h", "r"], as_index=False)
        .agg(mean_error=("error", "mean"))
    )

def compute_optimal_plot(diag_data: pd.DataFrame, diag_data_theta: pd.DataFrame):
    """Compute optimal r/a plots from averaged map and angle-dependent maps."""

    # Optimal curve from averaged map
    df_mean = prepare_mean_data(diag_data)

    idx = df_mean.groupby("norm_h")["mean_error"].idxmin()
    df_best = df_mean.loc[idx].sort_values("norm_h").reset_index(drop=True)

    x = np.log10(df_best["norm_h"].to_numpy())
    y = np.log10(df_best["r"].to_numpy())

    alpha, logC = np.polyfit(x, y, 1)
    C = 10 ** logC

    # Optimal curve for each theta
    best_theta_results = []

    for theta_rad in sorted(diag_data_theta["theta_rad"].unique()):
        df_theta = diag_data_theta[diag_data_theta["theta_rad"] == theta_rad]

        idx_theta = df_theta.groupby("norm_h")["error"].idxmin()
        df_best_theta = df_theta.loc[idx_theta].reset_index(drop=True)

        best_theta_results.append(df_best_theta)

    df_best_all_theta = pd.concat(best_theta_results, ignore_index=True)

    df_band = (
        df_best_all_theta
        .groupby("norm_h", as_index=False)
        .agg(
            r_mean=("r", "mean"),
            r_std=("r", "std"),
            r_min=("r", "min"),
            r_max=("r", "max"),
        )
        .sort_values("norm_h")
        .reset_index(drop=True)
    )

    df_band["r_upper"] = df_band["r_mean"] + df_band["r_std"]
    df_band["r_lower"] = df_band["r_mean"] - df_band["r_std"]

    x_band = np.log10(df_band["norm_h"].to_numpy())
    y_band = np.log10(df_band["r_mean"].to_numpy())

    alpha_band, logC_band = np.polyfit(x_band, y_band, 1)
    C_band = 10 ** logC_band

    return df_best, C, alpha, df_band, C_band, alpha_band

def plot_case(ax, diag_data, diag_data_theta, number: str):
    """Plot a optimal r-h scaling subplot."""

    df_best, C, alpha, df_band, C_band, alpha_band = compute_optimal_plot(
        diag_data,
        diag_data_theta,
    )

    norm_h_fit = np.linspace(0.001, 1.0, 999)
    r_fit = C * norm_h_fit ** alpha
    r_fit_band = C_band * norm_h_fit ** alpha_band

    h1 = ax.loglog(
        df_band["norm_h"],
        df_band["r_mean"],
        "o",
        color="tab:orange",
        markersize=3,
        label="Mean optimal $r/a$",
    )[0]

    h2 = ax.loglog(
        norm_h_fit,
        r_fit_band,
        "--",
        color="tab:orange",
        linewidth=1.0,
        label="Fitting (mean optimal $r/a$)",
    )[0]

    h3 = ax.fill_between(
        df_band["norm_h"],
        df_band["r_lower"],
        df_band["r_upper"],
        color="tab:orange",
        alpha=0.35,
        linewidth=0.0,
        label=r"$\pm 1$ STD of optimal $r/a$",
    )

    h4 = ax.loglog(
        df_best["norm_h"],
        df_best["r"],
        "o",
        color="tab:blue",
        markersize=3,
        label="Data (averaged map)",
    )[0]

    h5 = ax.loglog(
        norm_h_fit,
        r_fit,
        "--",
        color="tab:blue",
        linewidth=1.0,
        label="Fitting (averaged map)",
    )[0]

    ax.set_xlabel(r"$h/a$")
    ax.set_ylabel(r"$r/a$")

    ax.set_xlim(0.008, 1.0)
    ax.set_ylim(0.07, 1.0)

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
        -0.30,
        1.05,
        number,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )

    return [h1, h2, h3, h4, h5]


# Read data
case_files = [
    {
        "mean_csv": "data/sifconv/cache_modes/qfem/wm/map/map_wmo.csv",
        "theta_csv": "data/sifconv/cache_modes/qfem/wm/map/map_theta_wmo.csv",
    },
    {
        "mean_csv": "data/sifconv/cache_modes/qfem_qpe/wm/map/map_wmo.csv",
        "theta_csv": "data/sifconv/cache_modes/qfem_qpe/wm/map/map_theta_wmo.csv",
    },
    {
        "mean_csv": "data/sifconv/cache_modes/mpsa/wm/map/map_wmo.csv",
        "theta_csv": "data/sifconv/cache_modes/mpsa/wm/map/map_theta_wmo.csv",
    },
    {
        "mean_csv": "data/sifconv/cache_modes/qfem/wm/map/map_wms.csv",
        "theta_csv": "data/sifconv/cache_modes/qfem/wm/map/map_theta_wms.csv",
    },
    {
        "mean_csv": "data/sifconv/cache_modes/qfem_qpe/wm/map/map_wms.csv",
        "theta_csv": "data/sifconv/cache_modes/qfem_qpe/wm/map/map_theta_wms.csv",
    },
    {
        "mean_csv": "data/sifconv/cache_modes/mpsa/wm/map/map_wms.csv",
        "theta_csv": "data/sifconv/cache_modes/mpsa/wm/map/map_theta_wms.csv",
    },
]

# Plot six subfigures
fig, axes = plt.subplots(
    2,
    3,
    figsize=(6.8, 4.8),
    sharey=False,
)

axes_flat = axes.ravel()
legend_handles = None
numbers = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for ax, item, number in zip(axes_flat, case_files, numbers):
    diag_data, diag_data_theta = read_case(
        item["mean_csv"],
        item["theta_csv"],
    )

    handles = plot_case(
        ax=ax,
        diag_data=diag_data,
        diag_data_theta=diag_data_theta,
        number=number,
    )

    if legend_handles is None:
        legend_handles = handles

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
)

# Adjust layout to make room for the legend
# plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.15,
    top=0.95,
    wspace=0.40,
    hspace=0.30,
)

plt.savefig(
    os.path.join(save_fig, "Fig9_ScalingLaw_mixed.pdf"),
    format="pdf",
)

plt.savefig(
    os.path.join(save_fig, "Fig9_ScalingLaw_mixed.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.close(fig)
