"""
Utilities of self-convergence analysis for mixed-dimensional 
models by MPSA, and standard quadratic FEM (QFEM), and QFEM with
quarter-point elements (QFEM-QPE).
"""
from .quadrature_integration import QuadratureIntegration

from .overlap_p1_quad import (
    relative_error_p1_mpsa,
)

from .overlap_t6_quad import (
    relative_error_p2_fem,
)

from .overlap_fem_gaussian import (
    relative_error_p2_fem_qpe,
)

__all__ = [
    "QuadratureIntegration",
    "relative_error_p1_mpsa",
    "relative_error_p2_fem",
    "relative_error_p2_fem_qpe",
]