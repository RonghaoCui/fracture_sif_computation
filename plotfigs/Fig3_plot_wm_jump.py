from __future__ import annotations

import porepy as pp
import numpy as np

from westergaard2d import WestergaardMixedMode2d

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.image import imread
import os
save_dir = "figures/publication/Fig3/"
os.makedirs(save_dir, exist_ok=True)

a = 0.3
poi = 0.25
shear_modulus = 1.0
lam = 2 * shear_modulus * poi / (1 - 2 * poi)


params = {
    "prepare_simulation": True,
    "material_constants": {"solid": pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)},
    "a": a,
    "domain_size": 1.0,
    "meshing_arguments": {"cell_size": 0.025},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 45),
    "num_bem_segments": 1000,
    "poi": poi,
    "error_exclusion_zone_fracture_tips": 0.0,
    "sigma_inf": 1e-4,  # far_field normal stress
    "tau_inf": 1e-4,    # far_field shear stress
    "mu": 0.0,          # friction coefficient
    "p0": 0.0,          # This pressure value will not be calculated in this model. 
                        # But "p0" is needed because of model's parental class
}

model = WestergaardMixedMode2d(params=params)
pp.run_time_dependent_model(model)

# Save 2D map of displacement
# ux_map_path = os.path.join(save_dir, "ux_map_wm_135d.png")
# uy_map_path = os.path.join(save_dir, "uy_map_wm_135d.png")
# for g, data in model.mdg.subdomains(return_data=True):
#     # get 'u'（t=0 by default）
#     u = data.get(pp.TIME_STEP_SOLUTIONS, {}).get("u", {}).get(0, None)
#     if u is None:
#         continue

#     # get the displacement component along the x direction
#     x_disp = u[::2]
#     # get the displacement component along the y direction
#     y_disp = u[1::2]

#     # plot
#     pp.plot_grid(
#         g,
#         cell_value=x_disp,
#         fig_size=(3.2, 2.8),
#         plot_2d=True,
#         linewidth=0.3,
#     )
#     plt.tight_layout()
#     plt.savefig(
#         ux_map_path,
#         format="png",
#         dpi=600,
#         bbox_inches="tight",
#         pad_inches=0.02,
#     )
#     plt.close()

#     pp.plot_grid(
#         g,
#         cell_value=y_disp,
#         fig_size=(3.2, 2.8),
#         plot_2d=True,
#         linewidth=0.3,
#     )

#     plt.tight_layout()
#     plt.savefig(
#         uy_map_path,
#         format="png",
#         dpi=600,
#         bbox_inches="tight",
#         pad_inches=0.02,
#     )
#     plt.close()

# For displacement along the fracture
# Get the 1D fracture grid (assumes only one fracture)
fracture_grid = model.mdg.subdomains(dim=1)[0]
# x_coords = fracture_grid.cell_centers[0]  # X coordinate of each fracture cell center
theta = params["theta_rad"]
t_vec_frac = np.array([np.sin(theta), np.cos(theta)])  # Tangential vector along the fracture
coords = fracture_grid.cell_centers[:2, :]
tip_faces = fracture_grid.tags["tip_faces"].nonzero()[0]
tip_centers = fracture_grid.face_centers[:2, tip_faces]

s_tip = t_vec_frac @ tip_centers
tip0 = tip_centers[:, np.argmin(s_tip)]

rel_coords = coords - tip0[:, None]
x_plot = t_vec_frac @ rel_coords

# Get numerical normal/tangential displacement jump
u_jump = model.displacement_jump([fracture_grid])
u_jump_val = u_jump.value(model.equation_system)
normal_op = model.normal_component([fracture_grid])
u_open_numerical = (normal_op @ u_jump).value(model.equation_system)
tangent_op = model.tangential_component([fracture_grid])
# BUG: when the fracture angle is larger that 90 degree, the tangential jump is 
# observed to be negative. This is because of definition of the local basis in PorePy. 
# To fix this, we can simply add a negative sign to the tangential operator.
# This fracture orientation check is only conducted for the angle ranging from 0 to 180 degree. 
# If the angle is outside this range, the sign check should be conducted.
sign_t = 1.0 if np.sin(theta) >= 0 else -1.0
u_slip_numerical = sign_t * (tangent_op @ u_jump).value(model.equation_system)

u_open_analytical = model.exact_sol.exact_sol_normal_fracture(model.mdg)
u_slip_analytical = model.exact_sol.exact_sol_tangential_fracture(model.mdg)

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

# Set the scientific format
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))   

# ===== Combined 2x2 figure =====
fig, axes = plt.subplots(
    1, 2,
    figsize=(6.3, 2.7),   # width = 6.3 inches
    constrained_layout=True
)

# ax_ux, ax_uy = axes[0, 0], axes[0, 1]
ax_un, ax_ut = axes[0], axes[1]

# ---------- first row: maps ----------
# img_ux = imread(ux_map_path)
# img_uy = imread(uy_map_path)

# ax_ux.imshow(img_ux)
# ax_uy.imshow(img_uy)

# ax_ux.axis("off")
# ax_uy.axis("off")

# Plot the profile of normal displacement jump along the fracture
ax_un.plot(
    x_plot,
    u_open_analytical,
    '-',
    color='tab:blue',
    linewidth=1.5, 
    label="Analytical"
)

ax_un.plot(
    x_plot,
    u_open_numerical,
    'o',
    color='tab:orange',
    markersize=3, 
    label="Numerical"
)
ax_un.yaxis.set_major_formatter(formatter)
ax_un.set_xlabel("Distance from fracture tip 1")
ax_un.set_ylabel("Normal displacement jump")
ax_un.set_xlim(0.0, 0.60)
ax_un.set_ylim(1.0e-5, 5.0e-5)
ax_un.set_xticks(np.arange(0.0, 0.61, 0.1))
ax_un.set_yticks(np.arange(1.0e-5, 5.01e-5, 1.0e-5))
ax_un.grid(True, linestyle="--", alpha=0.5)
ax_un.legend(loc="upper left", frameon=False)

# Plot the profile of tangential displacement jump along the fracture
ax_ut.plot(
    x_plot,
    u_slip_analytical,
    '-',
    color='tab:blue',
    linewidth=1.5, 
    label="Analytical"
)
ax_ut.plot(
    x_plot,
    u_slip_numerical,
    'o',
    color='tab:orange',
    markersize=3, 
    label="Numerical"
)

ax_ut.yaxis.set_major_formatter(formatter)
ax_ut.set_xlabel("Distance from fracture tip 1")
ax_ut.set_ylabel("Tangential displacement jump")
ax_ut.set_xlim(0.0, 0.60)
ax_ut.set_ylim(1.0e-5, 5.0e-5)
ax_ut.set_xticks(np.arange(0.0, 0.61, 0.1))
ax_ut.set_yticks(np.arange(1.0e-5, 5.01e-5, 1.0e-5))
ax_ut.grid(True, linestyle="--", alpha=0.5)
ax_ut.legend(loc="upper left", frameon=False)

# ---------- panel labels ----------
# labels = ["(a)", "(b)"]
# for ax, label in zip(axes, labels):
#     ax.text(
#         0.02,
#         0.98,
#         label,
#         transform=ax.transAxes,
#         ha="left",
#         va="top",
#         fontsize=8,
#         fontweight="bold",
#         bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
#     )

fig.savefig(
    os.path.join(save_dir, "Fig3_wm_displacement_jump.pdf"),
    format="pdf",
    bbox_inches="tight",
)
fig.savefig(
    os.path.join(save_dir, "Fig3_wm_displacement_jump.png"),
    format="png",
    dpi=600,
    bbox_inches="tight",
)
plt.close(fig)
