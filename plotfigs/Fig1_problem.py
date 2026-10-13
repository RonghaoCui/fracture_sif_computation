"""Figure 1: The fractured domain with given external and internal
boundaries for the near-tip problem.
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

save_dir = "figures/publication/Fig1/"
os.makedirs(save_dir, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "lines.linewidth": 1.0,
        "mathtext.fontset": "dejavusans",
    }
)

fig = plt.figure(figsize=(3.5, 2.2))
ax = fig.add_axes([0, 0, 1, 1])

# Set limits and aspect ratio.
ax.set_xlim(0, 9.6)
ax.set_ylim(0, 6.0)
ax.set_aspect("equal")
ax.axis("off")

domain_color ="#f5dfcf"
line_color = "black"
guide_color = "0.45"

lw_boundary = 1.5
lw_fracture = 1.5
lw_inset = 2.5
lw_guide = 1.0

# Plot main domain.
domain_x = 0.15
domain_y = 0.15
domain_size = 5.0

# Fracture
fracture_length = 3.20
fracture_angle_deg = 45.0

# Circular enlargement
R = 1.8
circle_gap = 0.50
cy = 4.0

# Opening width in enlarged fracture
opening_gap = 0.80

# Dashed guide lines
source_fraction = 0.65
guide_angle_upper_deg = 135.0
guide_angle_lower_deg = 245.0

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
domain_right = domain_x + domain_size
domain_top = domain_y + domain_size

# External boundary label.
ax.text(
    domain_cx,
    domain_top + 0.25,
    r"$\Gamma_{D}$",
    ha="center",
    va="bottom",
)

# Fracture set-up.
# Direction along fracture:
#     t = (cos(theta), sin(theta))
# Normal direction:
#     n = (-sin(theta), cos(theta))
theta = math.radians(fracture_angle_deg)

tx = math.cos(theta)
ty = math.sin(theta)
nx = -math.sin(theta)
ny = math.cos(theta)

half_length = fracture_length / 2.0

fracture_start = (
    domain_cx - half_length * tx,
    domain_cy - half_length * ty,
)

fracture_end = (
    domain_cx + half_length * tx,
    domain_cy + half_length * ty,
)

ax.plot(
    [fracture_start[0], fracture_end[0]],
    [fracture_start[1], fracture_end[1]],
    color=line_color,
    linewidth=lw_fracture,
    solid_capstyle="butt",
    zorder=3,
)

# Circular plot.
cx = domain_right + circle_gap + R
circle = Circle(
    (cx, cy),
    R,
    facecolor=domain_color,
    edgecolor="none",
    zorder=2,
)
ax.add_patch(circle)

# Fracture surfaces in circular inset.
upper_cx = cx + 0.5 * opening_gap * nx
upper_cy = cy + 0.5 * opening_gap * ny
lower_cx = cx - 0.5 * opening_gap * nx
lower_cy = cy - 0.5 * opening_gap * ny

# Long enough to pass completely through the circle.
# Clipping removes everything outside the circle.
L = 2.5 * R

# White fracture opening.
p1 = (
    upper_cx - L * tx,
    upper_cy - L * ty,
)

p2 = (
    upper_cx + L * tx,
    upper_cy + L * ty,
)

p3 = (
    lower_cx + L * tx,
    lower_cy + L * ty,
)

p4 = (
    lower_cx - L * tx,
    lower_cy - L * ty,
)

fracture_opening = Polygon(
    [p1, p2, p3, p4],
    closed=True,
    facecolor="white",
    edgecolor="none",
    zorder=3,
)

ax.add_patch(fracture_opening)

# Clip the white opening to the circular inset.
fracture_opening.set_clip_path(circle)

# Thick fracture surfaces.
upper_line, = ax.plot(
    [
        upper_cx - L * tx,
        upper_cx + L * tx,
    ],
    [
        upper_cy - L * ty,
        upper_cy + L * ty,
    ],
    color=line_color,
    linewidth=lw_inset,
    solid_capstyle="butt",
    zorder=4,
)

lower_line, = ax.plot(
    [
        lower_cx - L * tx,
        lower_cx + L * tx,
    ],
    [
        lower_cy - L * ty,
        lower_cy + L * ty,
    ],
    color=line_color,
    linewidth=lw_inset,
    solid_capstyle="butt",
    zorder=4,
)

upper_line.set_clip_path(circle)
lower_line.set_clip_path(circle)

# Labels inside inset.
# Put labels along the normal direction, well outside the white opening.
label_offset = 0.62 * R

ax.text(
    cx + label_offset * nx,
    cy + label_offset * ny,
    r"$\Gamma_s^+$",
    ha="center",
    va="center",
    zorder=5,
)

ax.text(
    cx - label_offset * nx,
    cy - label_offset * ny,
    r"$\Gamma_s^-$",
    ha="center",
    va="center",
    zorder=5,
)

# Dashed guide lines
# Both start from exactly the same point on the fracture.
# Their endpoints are computed from the circle equation, so
# they lie exactly on the circular boundary.
# Point on the fracture used as the common source.
source = (
    fracture_start[0]
    + source_fraction * (fracture_end[0] - fracture_start[0]),
    fracture_start[1]
    + source_fraction * (fracture_end[1] - fracture_start[1]),
)

# Choose two positions on the left-hand side of the circle.
def point_on_circle(angle_deg: float) -> tuple[float, float]:
    """Return a point exactly on the circular boundary."""
    angle = math.radians(angle_deg)

    return (
        cx + R * math.cos(angle),
        cy + R * math.sin(angle),
    )

circle_upper = point_on_circle(guide_angle_upper_deg)
circle_lower = point_on_circle(guide_angle_lower_deg)

ax.plot(
    [source[0], circle_upper[0]],
    [source[1], circle_upper[1]],
    linestyle="--",
    color=guide_color,
    linewidth=lw_guide,
    zorder=2.5,
)

ax.plot(
    [source[0], circle_lower[0]],
    [source[1], circle_lower[1]],
    linestyle="--",
    color=guide_color,
    linewidth=lw_guide,
    zorder=2.5,
)

# Redraw circular boundary on top.
circle_outline = Circle(
    (cx, cy),
    R,
    facecolor="none",
    edgecolor=guide_color,
    linewidth=lw_guide,
    linestyle="--",
    zorder=6,
)

ax.add_patch(circle_outline)

# Save.
fig.savefig(
    os.path.join(save_dir, "Fig1_problem.pdf"),
    format="pdf",
)

fig.savefig(
    os.path.join(save_dir, "Fig1_problem.tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.close(fig)
