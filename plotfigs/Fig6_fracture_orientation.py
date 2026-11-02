r"""Figure 11: Illustration of fracture orientation in the domain, with
the fracture angle $\beta$ measured counterclockwise from the horizontal
reference. The fracture center is fixed at the domain center.
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

save_dir = "figures/publication/Fig6/"
os.makedirs(save_dir, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "lines.linewidth": 1.0,
        "mathtext.fontset": "dejavusans",
    }
)

fig = plt.figure(figsize=(3.8, 2.0))
ax = fig.add_axes([0, 0, 2.0 / 3.8, 1])

# Set limits and aspect ratio.
ax.set_xlim(-0.15, 5.45)
ax.set_ylim(-0.15, 5.45)
ax.set_aspect("equal")
ax.axis("off")

domain_color ="#f5dfcf"
line_color = "black"
guide_color = "0.45"

lw_boundary = 1.5
lw_fracture = 1.5
lw_guide = 1.0

# Plot main domain.
domain_x = 0.15
domain_y = 0.15
domain_size = 5.0

# Fracture.
fracture_length = 3.0
fracture_angle_deg = 35.0

# Main domain rectangle.
domain = Rectangle(
    (domain_x, domain_y),
    domain_size,
    domain_size,
    facecolor=domain_color,
    edgecolor=line_color,
    linewidth=lw_boundary,
    zorder=1,
)

ax.add_patch(domain)

domain_cx = domain_x + domain_size / 2.0
domain_cy = domain_y + domain_size / 2.0

# Counterclockwise angle from the horizontal direction.
beta = math.radians(fracture_angle_deg)
tx = math.cos(beta)
ty = math.sin(beta)

half_length = fracture_length / 2.0

fracture_start = (
    domain_cx - half_length * tx,
    domain_cy - half_length * ty,
)

fracture_end = (
    domain_cx + half_length * tx,
    domain_cy + half_length * ty,
)

rotated_line, = ax.plot(
    [fracture_start[0], fracture_end[0]],
    [fracture_start[1], fracture_end[1]],
    color=guide_color,
    linestyle="--",
    label="Rotated fracture",
    linewidth=lw_fracture,
    solid_capstyle="butt",
    zorder=3,
)

# Horizontal reference fracture.
reference_line, = ax.plot(
    [domain_cx - half_length, domain_cx + half_length],
    [domain_cy, domain_cy],
    label="Horizontal fracture",
    color=line_color,
    linewidth=lw_fracture,
    solid_capstyle="butt",
    zorder=3,
)

# Rotation center.
fracture_center, = ax.plot(
    domain_cx,
    domain_cy,
    "o",
    color=line_color,
    markersize=3,
    zorder=4,
    label="Fracture center"
)

# Counterclockwise angle.
arc_radius = 0.6
arc = Arc(
    (domain_cx, domain_cy),
    2.0 * arc_radius,
    2.0 * arc_radius,
    theta1=0.0,
    theta2=fracture_angle_deg,
    color=line_color,
    linewidth=lw_guide,
    zorder=4,
)
ax.add_patch(arc)

arrow_start = beta - math.radians(5.0)
ax.annotate(
    "",
    xy=(domain_cx + arc_radius * tx, domain_cy + arc_radius * ty),
    xytext=(
        domain_cx + arc_radius * math.cos(arrow_start),
        domain_cy + arc_radius * math.sin(arrow_start),
    ),
    arrowprops=dict(arrowstyle="->", color=line_color, lw=lw_guide,
                    mutation_scale=10, shrinkA=0, shrinkB=0),
    zorder=4,
)

label_radius = 0.9
ax.text(
    domain_cx + label_radius * math.cos(beta / 2.0),
    domain_cy + label_radius * math.sin(beta / 2.0),
    r"$\beta$",
    ha="center",
    va="center",
    fontsize=12,
    zorder=5,
)

ax.legend(
    handles=[reference_line, rotated_line, fracture_center],
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    borderaxespad=0,
    handlelength=2.8,
)

# Save.
fig.savefig(
    os.path.join(save_dir, "Fig6_fracture_orientation.pdf"),
    format="pdf",
)

fig.savefig(
    os.path.join(save_dir, "Fig6_fracture_orientation.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

fig.savefig(
    os.path.join(save_dir, "Fig6_fracture_orientation.png"),
    format="png",
    dpi=600,
)

plt.close(fig)
