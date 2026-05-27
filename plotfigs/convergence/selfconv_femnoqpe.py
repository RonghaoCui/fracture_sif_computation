from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import (
    LogLocator,
    LogFormatterMathtext,
)

import os
save_fig = "figures/convergence/selfconv/femnoqpe"
os.makedirs(save_fig, exist_ok=True)

# Read results from FEM without QPEs
# Read csv: Sneddon open mode
csv_path = "data/convergence/selfconv/femnoqpe/so"
df = pd.read_csv(os.path.join(csv_path, "selfconv_femnoqpe_so.csv"))

sqrt_dofs_so = np.sqrt(df["matrix_dof"].values)
matrix_errors_so = df["matrix_error_quad"].values
fracture_errors_so = df["fracture_error_quad"].values

# Read csv: Westergaard open mode
csv_path = "data/convergence/selfconv/femnoqpe/wo"
df = pd.read_csv(os.path.join(csv_path, "selfconv_femnoqpe_wo.csv"))

sqrt_dofs_wo = np.sqrt(df["matrix_dof"].values)
matrix_errors_wo = df["matrix_error_quad"].values
fracture_errors_wo = df["fracture_error_quad"].values

# Read csv: Westergaard shear mode
csv_path = "data/convergence/selfconv/femnoqpe/ws"
df = pd.read_csv(os.path.join(csv_path, "selfconv_femnoqpe_ws.csv"))

sqrt_dofs_ws = np.sqrt(df["matrix_dof"].values)
matrix_errors_ws = df["matrix_error_quad"].values
fracture_errors_ws = df["fracture_error_quad"].values

# Read csv: Westergaard mixed mode
csv_path = "data/convergence/selfconv/femnoqpe/wm"
df = pd.read_csv(os.path.join(csv_path, "selfconv_femnoqpe_wm.csv"))

sqrt_dofs_wm = np.sqrt(df["matrix_dof"].values)
matrix_errors_wm = df["matrix_error_quad"].values
fracture_errors_wm = df["fracture_error_quad"].values


# Compute slopes
slope_matrix_so, intercept_matrix_so = np.polyfit(
    np.log10(sqrt_dofs_so),
    np.log10(matrix_errors_so),
    1,
)

slope_fracture_so, intercept_fracture_so = np.polyfit(
    np.log10(sqrt_dofs_so),
    np.log10(fracture_errors_so),
    1,
)

slope_matrix_wo, intercept_matrix_wo = np.polyfit(
    np.log10(sqrt_dofs_wo),
    np.log10(matrix_errors_wo),
    1,
)

slope_fracture_wo, intercept_fracture_wo = np.polyfit(
    np.log10(sqrt_dofs_wo),
    np.log10(fracture_errors_wo),
    1,
)

slope_matrix_ws, intercept_matrix_ws = np.polyfit(
    np.log10(sqrt_dofs_ws),
    np.log10(matrix_errors_ws),
    1,
)

slope_fracture_ws, intercept_fracture_ws = np.polyfit(
    np.log10(sqrt_dofs_ws),
    np.log10(fracture_errors_ws),
    1,
)

slope_matrix_wm, intercept_matrix_wm = np.polyfit(
    np.log10(sqrt_dofs_wm),
    np.log10(matrix_errors_wm),
    1,
)

slope_fracture_wm, intercept_fracture_wm = np.polyfit(
    np.log10(sqrt_dofs_wm),
    np.log10(fracture_errors_wm),
    1,
)

print(f"Matrix slope (so):  {slope_matrix_so:.4f}")
print(f"Fracture slope (so): {slope_fracture_so:.4f}")

print(f"Matrix slope (wo):  {slope_matrix_wo:.4f}")
print(f"Fracture slope (wo): {slope_fracture_wo:.4f}")

print(f"Matrix slope (ws):  {slope_matrix_ws:.4f}")
print(f"Fracture slope (ws): {slope_fracture_ws:.4f}")

print(f"Matrix slope (wm):  {slope_matrix_wm:.4f}")
print(f"Fracture slope (wm): {slope_fracture_wm:.4f}")

# Matplotlib style
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 3.0,
    "mathtext.fontset": "dejavusans",
})

# Plot
fig, ax = plt.subplots(figsize=(3.5, 3.0))

ax.loglog(
    sqrt_dofs_so,
    matrix_errors_so,
    "-",
    color="tab:blue",
    label="Sneddon KI",
)

ax.loglog(
    sqrt_dofs_so,
    fracture_errors_so,
    "--",
    color="tab:blue",
    label="Sneddon KI",
)

ax.loglog(
    sqrt_dofs_wo,
    matrix_errors_wo,
    "-",
    color="tab:orange",
    label="Westergaard KI",
)

ax.loglog(
    sqrt_dofs_wo,
    fracture_errors_wo,
    "--",
    color="tab:orange",
    label="Westergaard KI",
)

ax.loglog(
    sqrt_dofs_ws,
    matrix_errors_ws,
    "-",
    color="tab:pink",
    label="Westergaard KII",
)

ax.loglog(
    sqrt_dofs_ws,
    fracture_errors_ws,
    "--",
    color="tab:pink",
    label="Westergaard KII",
)

ax.loglog(
    sqrt_dofs_wm,
    matrix_errors_wm,
    "-",
    color="tab:green",
    label="Westergaard mixed",
)

ax.loglog(
    sqrt_dofs_wm,
    fracture_errors_wm,
    "--",
    color="tab:green",
    label="Westergaard mixed",
)

# Reference triangle
x0 = 20
x1 = 30

y0 = 5e-3
y1 = y0 * (x0 / x1) ** 1.5   # slope = -1.5

# Triangle vertices
triangle_x = [x0, x0, x1]
triangle_y = [y0, y1, y1]

# Fill triangle
ax.fill(
    triangle_x,
    triangle_y,
    color="gray",
    alpha=0.5,
    zorder=1,
)

# Triangle edges
triangle_style = dict(
    color="black",
    linewidth=0.6,
    zorder=2,
)

ax.plot(
    [x0, x1],
    [y1, y1],
    **triangle_style,
)

ax.plot(
    [x0, x0],
    [y0, y1],
    **triangle_style,
)

ax.plot(
    [x0, x1],
    [y0, y1],
    **triangle_style,
)

# Labels
ax.text(
    x0 / 1.08,
    np.sqrt(y0 * y1),
    "1.5",
    ha="right",
    va="center",
)

ax.text(
    np.sqrt(x0 * x1),
    y1 / 1.08,
    "1.0",
    ha="center",
    va="top",
)

# Labels
ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
ax.set_ylabel("Relative error")

# Axis format
ax.set_xlim(10, 300)
ax.set_ylim(1e-3, 1.0)

ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Legend
# ax.legend(loc="upper right")

plt.tight_layout()

# Save figure
plt.savefig(
    os.path.join(save_fig, "selfconv_femnoqpe.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)

plt.close(fig)

