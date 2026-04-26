"""
Figure used for the CouFrac2026 abstract. 

This figure was plotted using the results from grids with auxilliary points. 
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext

import os
save_dir = "figures/auxpts/pub/abstract"
os.makedirs(save_dir, exist_ok=True)

# Nearest-tip sampling
# Sneddon problem (opening mode)
# MPSA
mpsa_dofs_open = np.array([
    16.0, 31.43246729, 61.28621378, 122.01639234, 244.03688246, 487.59614437
])

err_ave_mpsa_open = np.array([
    0.03532748, 0.02914586, 0.02020728, 0.01409485, 0.01642144, 0.00108328
]) 

# QFEM-QPE
femqpe_dofs_open = np.array([
    32.86335345, 63.7495098, 123.48279232, 244.94897428, 488.99079746
])

err_ave_femqpe_open = np.array([
    0.03577701, 0.04281079, 0.06531432, 0.03096159, 0.03012858
]) 

# QFEM
fem_dofs_open = np.array([
    32.86335345, 63.7495098, 123.48279232, 244.94897428, 488.99079746
])

err_ave_fem_open = np.array([
    0.14722644, 0.16201812, 0.22261816, 0.17039389, 0.17562448
]) 

# Westergaard problem (sliding mode)
# MPSA
mpsa_dofs_sliding = np.array([
    16.0, 31.43246729, 61.28621378, 122.01639234, 244.03688246, 487.59614437
])

err_ave_mpsa_sliding = np.array([
    0.31619868, 0.33475893, 0.36373258, 0.34144228, 0.34923245, 0.33763584
]) 

# QFEM-QPE
femqpe_dofs_sliding = np.array([
    32.86335345, 63.7495098, 123.48279232, 244.94897428, 488.99079746
])

err_ave_femqpe_sliding = np.array([
    0.00311614, 0.01066066, 0.02022317, 0.0089286, 0.009458
]) 

# QFEM
fem_dofs_sliding = np.array([
    32.86335345, 63.7495098, 123.48279232, 244.94897428, 488.99079746
])

err_ave_fem_sliding = np.array([
    0.27031229, 0.28781617, 0.29194237, 0.30881851, 0.3104813
]) 

# The scaling-based sampling
# Sneddon problem (opening mode)
# MPSA
mpsa_dofs_so_2 = np.array([
    16.32151574, 31.82437906, 62.97294016, 125.48940643, 250.10555251, 499.25663341
])

err_ave_mpsa_so_2 = np.array([
    0.01560495, 0.00494485, 0.00304355, 0.0023395, 0.00123267, 0.00078003
]) 

# QFEM-QPE
femqpe_dofs_so_2 = np.array([
     33.48983621, 64.52259576, 126.83203562, 251.86973768, 501.1059267 
])

err_ave_femqpe_so_2 = np.array([
    0.0094868, 0.00209868, 0.00053325, 0.00040623, 0.00024107
]) 


# QFEM no qpe
fem_dofs_so_2 = np.array([
    33.48983621, 64.52259576, 126.83203562, 251.86973768, 501.1059267
])

err_ave_fem_so_2 = np.array([
    0.00857427, 0.00148248, 0.00120427, 0.00049011, 0.00029562
]) 

# Westergaard problem (sliding mode)
# MPSA
mpsa_dofs_ws_2 = np.array([
    16.32151574, 31.82437906, 62.97294016, 125.48940643, 250.10555251, 499.25663341
])

err_ave_mpsa_ws_2 = np.array([
    0.13459652, 0.08088968, 0.04949771, 0.03009739, 0.01899039, 0.01155998
]) 

# QFEM-QPE
femqpe_dofs_ws_2 = np.array([
    33.48983621, 64.52259576, 126.83203562, 251.86973768, 501.1059267 
])

err_ave_femqpe_ws_2 = np.array([
    0.00311357, 0.00672516, 0.00166874, 0.0006794, 0.00023399
]) 


# QFEM
fem_dofs_ws_2 = np.array([
    33.48983621, 64.52259576, 126.83203562, 251.86973768, 501.1059267
])

err_ave_fem_ws_2 = np.array([
    0.012944, 0.00190451, 0.00090276, 0.00066994, 0.0003191
]) 

# Reference line
num_dofs_line = np.array([10, 30, 50, 70, 90, 100, 300, 500, 1000])
initial_value_line = 1e-3
first_order_line = initial_value_line * (num_dofs_line[0] / num_dofs_line)
half_order_line = initial_value_line  * (num_dofs_line[0] / num_dofs_line) ** 0.5

# Plot
mpl.rcParams.update({
    "font.family": "serif",     
    "font.size": 6.0,             
    "axes.labelsize": 6.0,       
    "axes.titlesize": 6.0,        
    "legend.fontsize": 6.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,  
    "lines.linewidth": 1.0,      
    "lines.markersize": 2.5,  
    "mathtext.fontset": "dejavusans",  
})
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))   # Scientific form

fig, ax = plt.subplots(1, 2, figsize=(4.60, 1.65))   # inches
# Plotting the results for the nearest-tip sampling
# Sneddon problem (opening mode)
# MPSA
ax[0].loglog(
    mpsa_dofs_open,
    err_ave_mpsa_open,
    'x--',
    color='tab:blue',
    label="MPSA (opening)"
)

# QFEM-QPE
ax[0].loglog(
    femqpe_dofs_open,
    err_ave_femqpe_open,
    'x--',
    color='tab:orange',
    label="QFEM-QPE (opening)"
)

# QFEM
ax[0].loglog(
    fem_dofs_open,
    err_ave_fem_open,
    'x--',
    color='tab:pink',
    label="QFEM (opening)"
)

# Westergaard problem (sliding mode)
# MPSA
ax[0].loglog(
    mpsa_dofs_sliding,
    err_ave_mpsa_sliding,
    'o:',
    color='tab:blue',
    label="MPSA (sliding)"
)

# QFEM-QPE
ax[0].loglog(
    femqpe_dofs_sliding,
    err_ave_femqpe_sliding,
    'o:',
    color='tab:orange',
    label="QFEM-QPE (sliding)"
)

# QFEM
ax[0].loglog(
    fem_dofs_sliding,
    err_ave_fem_sliding,
    'o:',
    color='tab:pink',
    label="QFEM (sliding)"
)

ax[0].xaxis.set_major_formatter(formatter)
ax[0].yaxis.set_major_formatter(formatter)
ax[0].set_xlim(10, 700)
ax[0].set_ylim(0.00005, 1.0)
ax[0].xaxis.set_major_locator(LogLocator(base=10))
ax[0].xaxis.set_major_formatter(LogFormatterMathtext())
ax[0].xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax[0].yaxis.set_major_locator(LogLocator(base=10))
ax[0].yaxis.set_major_formatter(LogFormatterMathtext())
ax[0].yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax[0].set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
ax[0].set_ylabel("Relative error")

# ax[0].legend(loc="lower left", fontsize=4.0, ncol=2)
ax[0].grid(True, which="both", linestyle="--", alpha=0.3)

# Plotting the results for the scaling-based sampling
# Sneddon problem (opening mode)
# MPSA
ax[1].loglog(
    mpsa_dofs_so_2,
    err_ave_mpsa_so_2,
    'x--',
    color='tab:blue',
    label="MPSA (opening)"
)

# QFEM-QPE
ax[1].loglog(
    femqpe_dofs_so_2,
    err_ave_femqpe_so_2,
    'x--',
    color='tab:orange',
    label="QFEM-QPE (opening)"
)

# QFEM
ax[1].loglog(
    fem_dofs_so_2,
    err_ave_fem_so_2,
    'x--',
    color='tab:pink',
    label="QFEM (opening)"
)

# Westergaard problem (sliding mode)
# MPSA
ax[1].loglog(
    mpsa_dofs_ws_2,
    err_ave_mpsa_ws_2,
    'o:',
    color='tab:blue',
    label="MPSA (sliding)"
)

# QFEM-QPE
ax[1].loglog(
    femqpe_dofs_ws_2,
    err_ave_femqpe_ws_2,
    'o:',
    color='tab:orange',
    label="QFEM-QPE (sliding)"
)

# QFEM
ax[1].loglog(
    fem_dofs_ws_2,
    err_ave_fem_ws_2,
    'o:',
    color='tab:pink',
    label="QFEM (sliding)"
)

ax[1].loglog(
    num_dofs_line, 
    first_order_line,
    '--', color='k',
    linewidth=1.0,
    label="Reference slope=-1"
)

ax[1].xaxis.set_major_formatter(formatter)
ax[1].yaxis.set_major_formatter(formatter)
ax[1].set_xlim(10, 700)
ax[1].set_ylim(0.00005, 1.0)
ax[1].xaxis.set_major_locator(LogLocator(base=10))
ax[1].xaxis.set_major_formatter(LogFormatterMathtext())
ax[1].xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax[1].yaxis.set_major_locator(LogLocator(base=10))
ax[1].yaxis.set_major_formatter(LogFormatterMathtext())
ax[1].yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

ax[1].set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
ax[1].set_ylabel("Relative error")

# ax[1].legend(loc="lower left", fontsize=6.5, ncol=2)
ax[1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  
    frameon=False,
    fontsize=6.0,
)
ax[1].grid(True, which="both", linestyle="--", alpha=0.3)

fig.subplots_adjust(
    left=0.40,   
    right=0.80,
    bottom=0.25,
    top=0.92,
    wspace=0.3
)

# ax[0].text(
#     -0.40, 1.05, "(a)",
#     transform=ax[0].transAxes,
#     fontsize=6.5,
#     # fontweight="bold",
#     va="top"
# )

# ax[1].text(
#     -0.40, 1.05, "(b)",
#     transform=ax[1].transAxes,
#     fontsize=6.5,
#     # fontweight="bold",
#     va="top"
# )

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "CouFrac2026_abstract.tiff"),
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)

