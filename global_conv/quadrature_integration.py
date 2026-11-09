from __future__ import annotations

import numpy as np
import porepy as pp

from fems import mmfa


class QuadratureIntegration:
    """Quadrature-based integration utilities for global convergence analysis."""

    @staticmethod
    def triangle_area(xy: np.ndarray) -> float:
        """Compute the area of a triangle given its vertices.

        Parameters:
            xy: shape = (2, 3), where each column is a vertex (x, y).

        Returns:
            Area of the triangle.
        """
        p0 = xy[:, 0]
        p1 = xy[:, 1]
        p2 = xy[:, 2]

        return 0.5 * abs(np.linalg.det(np.column_stack((p1 - p0, p2 - p0))))

    @staticmethod
    def triangle_quadrature_degree4(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the degree-4 quadrature rule on one physical triangle.

        Proposed by Dunavant (1985), this method offers 6-point quadrature consisting
        of two groups on a reference triangle. Each group consists of 3 quadrature
        points, resulting in a total of 6 quadrature points. In each group, the quadrature
        points have the same weight and a format of barycentric coordinates, (a, a, b),
        where a + a + b = 1.

        The reference triangle is defined by vertices (0,0), (1,0), and (0,1), whose area
        is 0.5. Therefore, the quadrature weights of six quadrature points sum to 0.5 on
        the reference triangle. This is the reason why we at first halve the weights in the
        Dunavant (1985) and then multiply the weights by 2.0 * physical area to get the
        correct quadrature weights on the physical triangle.

        The reference can be found in:
            - Dunavant, D. (1985). International Journal for Numerical Methods in
            Engineering, 21(6), 1129-1148.

        Parameters:
            xy: triangle vertices, shape = (2, 3).

        Returns:
            quad_x: quadrature points, shape = (2, num_quad).
            quad_w: weights, shape = (num_quad,).
        """
        # Dunavant 6-point rule, degree 4.
        a1 = 0.445948490915965
        b1 = 0.108103018168070
        w1 = 0.223381589678011 / 2.0

        a2 = 0.091576213509771
        b2 = 0.816847572980459

        w2 = 0.109951743655322 / 2.0

        bary = np.array(
            [
                [a1, a1, b1],
                [a1, b1, a1],
                [b1, a1, a1],
                [a2, a2, b2],
                [a2, b2, a2],
                [b2, a2, a2],
            ]
        )

        weight_ref = np.array([w1, w1, w1, w2, w2, w2])

        area = QuadratureIntegration.triangle_area(xy)
        quad_x = xy @ bary.T

        # Dunavant weights sum to 0.5 on the reference triangle.
        # Therefore multiply by 2 * physical area.
        quad_w = 2.0 * area * weight_ref

        return quad_x, quad_w
    
    @staticmethod
    def transfer_triangles(tr: pp.Grid) -> np.ndarray:
        """Get node indices for each triangle cell in the transfer grid.

        Parameters:
            tr: transfer grid, which is a 2D simplex grid.

        Returns:
            node indices: cell_id -> [node0, node1, node2],
            shape = (num_cells, 3).
        """
        cn = tr.cell_nodes().tocsc()
        return cn.indices.reshape((3, tr.num_cells), order="F").T

    @staticmethod
    def segment_quadrature_gauss(
        s0: float,
        s1: float,
        num_quad: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Gauss-Legendre quadrature rule on one 1D segment [s0, s1].

        Here, we use the standard Gauss-Legendre quadrature, which is exact 
        for polynomials of degree up to 2*num_quad - 1. Therefore, in order to cover
        the P1 and P2 fields (whose squared errors have degrees of 2 and 4, repectively), 
        num_quad = 3 or higher order can be used.

        Returns: 
            quad_s: quadrature points, shape = (num_quad,).
            quad_w: weights, shape = (num_quad,).
        """
        # position_ref, weight_ref = np.polynomial.legendre.leggauss(num_quad)
        position_ref, weight_ref = mmfa.gausspoint(num_quad, '1D')

        mid_pts = 0.5 * (s0 + s1)
        half_length = 0.5 * (s1 - s0)

        quad_s = mid_pts + half_length * position_ref.ravel()
        quad_w = abs(half_length) * weight_ref.ravel()

        return quad_s, quad_w
    
    @staticmethod
    def triangle_quadrature_gaussian(
        xy: np.ndarray,
        num_quad: int, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triangle quadrature on one physical triangle using Gaussian quadrature. 

        
        """
        position_ref, weight_ref = mmfa.gausspoint(num_quad, 'T3')

        xi = position_ref[:, 0]
        eta = position_ref[:, 1]

        # Mapping from reference triangle to physical triangle. 
        quad_x = (
            xy[:, [0]]
            + (xy[:, [1]] - xy[:, [0]]) * xi[None, :]
            + (xy[:, [2]] - xy[:, [0]]) * eta[None, :]
        )
        
        area = QuadratureIntegration.triangle_area(xy)

        quad_w = 2.0 * area * weight_ref.ravel()

        return quad_x, quad_w

    
