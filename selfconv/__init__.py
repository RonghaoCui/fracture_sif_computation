"""
Self-convergence utilities for mixed-dimensional MPSA analysis.
"""
from .quadrature_integration import QuadratureIntegration

from .overlap_p1 import (
    matrix_overlap_l2_components_p1,
    fracture_overlap_l2_components_p1,
    interface_overlap_l2_components_p1,
)

from .overlap_p1_quad import (
    relative_error_p1_mpsa,
)

from .overlap_t6_quad import (
    relative_error_p2_fem,
)

__all__ = [
    "QuadratureIntegration",
    "matrix_overlap_l2_components_p1",
    "fracture_overlap_l2_components_p1",
    "interface_overlap_l2_components_p1",
    "relative_error_p1_mpsa",
    "relative_error_p2_fem",
]