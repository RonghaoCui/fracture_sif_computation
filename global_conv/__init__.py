"""
Utilities of convergence analysis for mixed-dimensional models by MPSA, and
standard quadratic FEM (QFEM), and QFEM with quarter-point elements (QFEM-QPE).
"""
from .quadrature_integration import QuadratureIntegration

from .reference_p1_quad import (
    relative_ref_error_p1_mpsa,
)

from .reference_fem_gaussian import (
    relative_ref_error_p2_fem_qpe,
)

__all__ = [
    "QuadratureIntegration",
    "relative_ref_error_p1_mpsa",
    "relative_ref_error_p2_fem_qpe",
]
