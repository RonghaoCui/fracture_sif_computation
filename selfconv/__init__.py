"""
Self-convergence utilities for mixed-dimensional MPSA analysis.
"""

from .overlap_p1 import (
    matrix_overlap_l2_components_p1,
    fracture_overlap_l2_components_p1,
    interface_overlap_l2_components_p1,
)

from .overlap_p1_quad import (
    relative_l2_error_p1_field_between_models,
)

from .overlap_t6_quad import (
    matrix_overlap_l2_components_t6_quad,
    fracture_overlap_l2_components_p2_quad,
)

__all__ = [
    "matrix_overlap_l2_components_p1",
    "fracture_overlap_l2_components_p1",
    "interface_overlap_l2_components_p1",
    "relative_l2_error_p1_field_between_models",
    "matrix_overlap_l2_components_t6_quad",
    "fracture_overlap_l2_components_p2_quad",
]