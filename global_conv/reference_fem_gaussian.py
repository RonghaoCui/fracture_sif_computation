"""
Convergence functionalities of P2 fields for both QFEM and QFEM-QPE models containing
a fracture in a 2D domain.

The QFEM models are solved with the standard T6 elements, where the mid-side nodes are
placed at the midpoints of the edges.

The QFEM-QPE models are solved with the standard T6 elements and QPEs represented by
collapsed Q8 elements, where the mid-side nodes in QPEs are shifted to the location of
quarter points along the edges in the original 2D grid.

The reference solutions on 2D grids and across the 1D fracture are determined by the
functions implemented for each benchmark.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from global_conv import QuadratureIntegration

def _barycentric_coordinates(
    vertices: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates of given points in a triangle.

    For any point p in the triangle, its barycentric coordinates (L0, L1, L2) satisfy
    p = L0·v0 + L1·v1 + L2·v2, and L0 + L1 + L2 = 1.

    Parameters:
        vertices: triangle vertices, shape = (2, 3), ordered as [v0, v1, v2].
        points: points, shape = (2, num_points).

    Returns:
        barycentric coordinates of given points, shape = (3, num_points), corresponding 
            to [v0, v1, v2].
    """
    A = np.vstack((vertices, np.ones(3)))
    rhs = np.vstack((points, np.ones(points.shape[1])))
    return np.linalg.solve(A, rhs)


def _t6_shape_functions_2d(L: np.ndarray) -> np.ndarray:
    """Compute T6 shape functions from barycentric coordinates of given points.

    The T6 node order in a triangle is consistent with the FEM mesh 
    construction: [v0, m01, v1, m12, v2, m02]. Correspondingly, the T6 shape 
    functions for these six nodes are defined as follows, where L0, L1, L2 are barycentric 
    coordinates of given points, respectively:
        N0 = L0 * (2.0*L0 - 1.0),
        N1 = 4.0 * L0 * L1,
        N2 = L1 * (2.0*L1 - 1.0),
        N3 = 4.0 * L1 * L2,
        N4 = L2 * (2.0*L2 - 1.0),
        N5 = 4.0 * L0 * L2.

    Parameters:
        L: shape = (3, num_points)
            barycentric coordinates, ordered as [L0, L1, L2] for vertices [v0, v1, v2].

    Returns:
        N: shape = (6, num_points)
            values of T6 shape functions at given points. 
    """
    L0 = L[0]
    L1 = L[1]
    L2 = L[2]

    return np.vstack(
        (
            L0 * (2.0 * L0 - 1.0),
            4.0 * L0 * L1,
            L1 * (2.0 * L1 - 1.0),
            4.0 * L1 * L2,
            L2 * (2.0 * L2 - 1.0),
            4.0 * L0 * L2,
        )
    )


def _fem_displacement_by_node(U: np.ndarray) -> np.ndarray:
    """Convert flattened FEM displacement (2*num_nodes,) 
    to shape = (2, num_nodes).
    """
    return np.vstack((U[0::2], U[1::2]))


def _eval_t6_2d(
    quad_x: np.ndarray,
    element_nodes: np.ndarray,
    element_u: np.ndarray,
) -> np.ndarray:
    """Compute a standard quadratic (P2) FEM displacement field at given points in 2D.

    This method assumes a standard T6 element, where the physical triangle is 
    represented by the corner vertices [v0, v1, v2] as indicated in ``element_nodes``. 
    This is ONLY valid for the standard T6 element without QPE, where the mid-side nodes 
    are correctly placed at the midpoints of the edges in the original 2D grid.

    Parameters: 
        quad_x: shape = (2, num_points),
            quadrature points in the same 2D frame as element_nodes.
        element_nodes: shape = (6, 2), 
            coordinates of the 6 T6 nodes, ordered as [v0, m01, v1, m12, v2, m02].
        element_u: shape = (2, 6),
            displacement values at the 6 T6 nodes. 

    Returns:
        displacement at given quadrature points, shape = (2, num_points).
    """
    vertices = element_nodes[[0, 2, 4], :].T
    # Compute barycentric coordinates of quadrature points in the triangle 
    # defined by the corner vertices.
    L = _barycentric_coordinates(vertices, quad_x)
    # Compute T6 shape functions at quadrature points.
    N = _t6_shape_functions_2d(L)
    return element_u @ N


def _q8_shape_functions_and_derivatives(
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q8 serendipity shape functions and derivatives.

    Node order follows your mmfa.Q8element:
        0: (-1,-1), 1: (0,-1), 2: (1,-1), 3: (1,0),
        4: (1,1),   5: (0,1),  6: (-1,1), 7: (-1,0)
    """
    corx = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=float)
    cory = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=float)

    N = np.zeros(8)
    dN_dxi = np.zeros(8)
    dN_deta = np.zeros(8)

    for i in range(8):
        xisi = corx[i]
        etai = cory[i]

        N[i] = (
            ((1.0 + xi * xisi) * (1.0 + eta * etai)
             - (1.0 - xi**2) * (1.0 + eta * etai)
             - (1.0 - eta**2) * (1.0 + xi * xisi))
            * xisi**2 * etai**2 / 4.0
            + (1.0 - xi**2) * (1.0 + eta * etai)
            * (1.0 - xisi**2) * etai**2 / 2.0
            + (1.0 - eta**2) * (1.0 + xi * xisi)
            * (1.0 - etai**2) * xisi**2 / 2.0
        )

        dN_dxi[i] = (
            (xisi * (1.0 + eta * etai)
             + 2.0 * xi * (1.0 + eta * etai)
             - xisi * (1.0 - eta**2))
            * xisi**2 * etai**2 / 4.0
            - 2.0 * xi * (1.0 + eta * etai)
            * (1.0 - xisi**2) * etai**2 / 2.0
            + xisi * (1.0 - eta**2)
            * (1.0 - etai**2) * xisi**2 / 2.0
        )

        dN_deta[i] = (
            (etai * (1.0 + xi * xisi)
             - etai * (1.0 - xi**2)
             + 2.0 * eta * (1.0 + xi * xisi))
            * xisi**2 * etai**2 / 4.0
            + etai * (1.0 - xi**2)
            * (1.0 - xisi**2) * etai**2 / 2.0
            - 2.0 * eta * (1.0 + xi * xisi)
            * (1.0 - etai**2) * xisi**2 / 2.0
        )

    return N, dN_dxi, dN_deta


def _collapsed_q8_shape_functions_and_derivatives(
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse Q8 shape functions to the six-node QPE representation.

    This matches your stiffness assembly:
        M0 = N0 + N6 + N7
        M1 = N1
        M2 = N2
        M3 = N3
        M4 = N4
        M5 = N5
    """
    N8, dN8_dxi, dN8_deta = _q8_shape_functions_and_derivatives(xi, eta)

    M = np.array([
        N8[0] + N8[6] + N8[7],
        N8[1],
        N8[2],
        N8[3],
        N8[4],
        N8[5],
    ])

    dM_dxi = np.array([
        dN8_dxi[0] + dN8_dxi[6] + dN8_dxi[7],
        dN8_dxi[1],
        dN8_dxi[2],
        dN8_dxi[3],
        dN8_dxi[4],
        dN8_dxi[5],
    ])

    dM_deta = np.array([
        dN8_deta[0] + dN8_deta[6] + dN8_deta[7],
        dN8_deta[1],
        dN8_deta[2],
        dN8_deta[3],
        dN8_deta[4],
        dN8_deta[5],
    ])

    return M, dM_dxi, dM_deta


def _inverse_collapsed_q8_coordinates(
    x_phys: np.ndarray,
    element_nodes: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> tuple[float, float]:
    """Find Q8 reference coordinates (xi, eta) for one physical point.

    This is the inverse of the collapsed-Q8 geometry mapping used by QPE.
    """
    # Use standard T6 barycentric coordinates as an initial guess.
    vertices = element_nodes[[0, 2, 4], :].T
    L0, L1, L2 = _barycentric_coordinates(
        vertices,
        x_phys.reshape(2, 1),
    ).ravel()

    # Map triangle barycentric guess to a rough Q8-square guess.
    # This is not exact, but usually good enough for Newton.
    xi = 2.0 * L1 + L2 - 1.0
    eta = L2 - L0

    for _ in range(max_iter):
        M, dM_dxi, dM_deta = _collapsed_q8_shape_functions_and_derivatives(
            xi,
            eta,
        )

        x_map = element_nodes.T @ M
        residual = x_map - x_phys

        if np.linalg.norm(residual) < tol:
            # print(f"Converged in {_} iterations with residual norm {np.linalg.norm(residual):.2e}")
            return xi, eta

        dx_dxi = element_nodes.T @ dM_dxi
        dx_deta = element_nodes.T @ dM_deta

        J = np.column_stack((dx_dxi, dx_deta))

        delta = np.linalg.solve(J, residual)

        xi -= float(delta[0])
        eta -= float(delta[1])

    raise RuntimeError(
        "Newton failed in inverse collapsed-Q8 mapping."
    )


def _eval_qpe_collapsed_q8_2d(
    quad_x: np.ndarray,
    element_nodes: np.ndarray,
    element_u: np.ndarray,
) -> np.ndarray:
    """Evaluate QPE displacement using the same collapsed-Q8 basis as assembly.

    Parameters: 
        quad_x: physical quadrature points, shape = (2, num_points).
        element_nodes: six QPE/T6 node coordinates, shape = (6, 2).
        element_u: six nodal displacement values, shape = (2, 6).

    Returns:
        uq: displacement at quadrature points, shape = (2, num_points).
    """
    uq = np.zeros((2, quad_x.shape[1]), dtype=float)

    for k in range(quad_x.shape[1]):
        xi, eta = _inverse_collapsed_q8_coordinates(
            x_phys=quad_x[:, k],
            element_nodes=element_nodes,
        )

        M, _, _ = _collapsed_q8_shape_functions_and_derivatives(xi, eta)
        uq[:, k] = element_u @ M

    return uq


def _qpe_cell_set(model: object) -> set[int]:
    """Return cell indices marked as QPE."""
    qpe_cells: list[int] = []

    for cells in model.elements_quarter:
        if len(cells) > 0:
            qpe_cells.extend(np.asarray(cells, dtype=int).tolist())

    return set(np.unique(qpe_cells).tolist())


def matrix_error_integration_t6_quad(
    model: object,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    num_quad: int = 7,
    use_qpe: bool = False,
) -> tuple[float, float]:
    """Compute squared L2 error integrals for T6 FEM fields on 2D grids using the
    quadrature rule, based on the given reference solution.

    It computes: 
        numerator_sq   = ∫_Omega ||u_num - u_ref||^2 dx
        denominator_sq = ∫_Omega ||u_ref||^2 dx
    where the numerical field and the reference field are on the same 2D grid.

    Parameters: 
        model: a solved FEM model.
        reference_solution: reference solution function by accepting coordinates with 
            shape = (2, n_points) and returning displacement components with
            shape = (n_components, n_points).
        num_quad: parameter of the Gaussian quadrature rule.
        use_qpe: whether the model uses quarter-point elements.

    Returns: 
        numerator_sq: squared L2 error integral over the matrix domain.
        denominator_sq: squared L2 norm of the reference solution over the matrix domain.
    """
    nodes = model.nodes_p6
    elements = model.elements_t6
    u_node = _fem_displacement_by_node(model.U)

    qpe_elems = _qpe_cell_set(model) if use_qpe else set()

    numerator_sq = 0.0
    denominator_sq = 0.0
    for cell_id, tri_nodes in enumerate(elements):
        element_nodes = nodes[tri_nodes,:2]
        element_u = u_node[:,tri_nodes]

        # Corner vertices in the T6 node ordering.
        vertices = element_nodes[[0, 2, 4],:].T

        quad_x, quad_w = QuadratureIntegration.triangle_quadrature_gaussian(
            vertices,
            num_quad
        )

        if cell_id in qpe_elems:
            u_num_q = _eval_qpe_collapsed_q8_2d(
                quad_x=quad_x,
                element_nodes=element_nodes,
                element_u=element_u,
            )
        else:
            u_num_q = _eval_t6_2d(
                quad_x=quad_x,
                element_nodes=element_nodes,
                element_u=element_u,
            )

        u_ref_q = reference_solution(quad_x)
        diff_q = u_num_q - u_ref_q
        numerator_sq += float(np.sum(quad_w * np.sum(diff_q**2, axis=0)))
        denominator_sq += float(np.sum(quad_w * np.sum(u_ref_q**2, axis=0)))

    return numerator_sq, denominator_sq


def _p2_shape_functions_and_derivatives_1d(
    xi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """P2 shape functions and derivatives for 1D segments.
    """
    N = np.vstack((
        0.5 * xi * (xi - 1.0),
        1.0 - xi**2,
        0.5 * xi * (xi + 1.0),
    ))

    dN_dxi = np.vstack((
        xi - 0.5,
        -2.0 * xi,
        xi + 0.5,
    ))

    return N, dN_dxi


def _fracture_p2_jump_segments_from_model(
    model: object,
    fracture_center: np.ndarray,
    tangent: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Obtain paired fracture edges and nodal displacement jumps on one physical 1D segment.

    Fracture orientation is considered here by projecting fracture nodes onto the 
    tangential direction of the fracture. This produces 1D coordinates along the 
    fracture, used for sorting the three nodes on each quadratic edge in the 
    order: left, middle, right. 

    Parameters: 
        model: a solved FEM model.
        fracture_center: 2D coordinates of the fracture center, shape = (2,).
        tangent: Unit tangent defining the positive direction of the fracture,
            shape = (2,).

    Returns:
        segs: the list of fracture P2 segments. 
            Each segment is represented as (s, jump_local),
            where 
            s: 1D coordinates of nodes relative to the fracture center ordered along the
                tangent, shape = (3,).
            jump_local: local displacement jump, shape = (2,3).
    """
    normal = np.array([-tangent[1], tangent[0]])
    u_node = _fem_displacement_by_node(model.U)

    # [n0, m01, n1] for each edge. 
    upper_edges = model.upper_frac_edge_dofs
    lower_edges = model.lower_frac_edge_dofs

    segs = []

    for upper, lower in zip(upper_edges, lower_edges):
        
        nodes = model.nodes_p6[upper, :2]
        s = (nodes - fracture_center) @ tangent

        # 1D coordinates along the fracture, used for sorting 
        # the points in the correct order.
        idx = np.argsort(s)

        s = s[idx]
        upper = upper[idx]
        lower = lower[idx]

        jump = u_node[:, upper] - u_node[:, lower]

        jump_local = np.vstack((
            tangent@jump,
            normal@jump,
        )) 

        segs.append((s, jump_local))

    return segs


def fracture_error_integration_p2_quad(
    model: object,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    fracture_center: np.ndarray,
    tangent: np.ndarray,
    num_quad: int = 7,
) -> tuple[float, float]:
    """Compute squared L2 error integrals for the P2 field on 1D grids using the quadrature
    rule, based on the given reference solution.

    It computes: 
        numerator_sq   = ∫_Gamma ||u_num - u_ref||^2 ds
        denominator_sq = ∫_Gamma ||u_ref||^2 ds

    Parameters: 
        model: a solved FEM model.
        fracture_center: 2D coordinates of the fracture center, shape = (2,).
        tangent: Unit tangent defining the positive direction of the fracture,
            shape = (2,).
        num_quad: parameter of the Gaussian quadrature rule.
    """
    segs = _fracture_p2_jump_segments_from_model(
        model,
        fracture_center = fracture_center,
        tangent = tangent,
    )

    xi, w_ref = QuadratureIntegration.segment_quadrature_gauss(
            -1.0, 1.0, num_quad
    )

    N, dN_dxi = _p2_shape_functions_and_derivatives_1d(xi)

    numerator_sq = 0.0
    denominator_sq = 0.0

    for s_seg, s_jump in segs:
        # Map reference points to physical coordinates. 
        s_quad = s_seg @ N
        ds_dxi = s_seg @ dN_dxi
        w_quad = w_ref * np.abs(ds_dxi)

        jump_num_q = s_jump @ N
        jump_ref_q = reference_solution(s_quad)

        diff_quad = jump_num_q - jump_ref_q

        numerator_sq += float(np.sum(w_quad * np.sum(diff_quad**2, axis=0)))
        denominator_sq += float(np.sum(w_quad * np.sum(jump_ref_q**2, axis=0)))

    return numerator_sq, denominator_sq


def relative_ref_error_p2_fem_qpe(
    model: object,
    reference_solution_2d: Callable[[np.ndarray], np.ndarray],
    reference_solution_1d: Callable[[np.ndarray], np.ndarray],
    fracture_center: np.ndarray,
    tangent: np.ndarray,
    use_qpe: bool = False,
    num_quad_matrix: int = 7,
    num_quad_fracture: int = 7,
) -> tuple[float, float]:
    """Return relative L2 error of matrix displacement and fracture displacement jump.
    """

    matrix_num_sq, matrix_den_sq = matrix_error_integration_t6_quad(
        model=model,
        reference_solution=reference_solution_2d,
        num_quad=num_quad_matrix,
        use_qpe=use_qpe,
    )

    # QPE does not need to be specified in 1D.
    fracture_num_sq, fracture_den_sq = fracture_error_integration_p2_quad(
        model=model,
        reference_solution=reference_solution_1d,
        fracture_center=fracture_center,
        tangent=tangent,
        num_quad=num_quad_fracture,
    )

    matrix_error = np.sqrt(matrix_num_sq / matrix_den_sq)
    fracture_error = np.sqrt(fracture_num_sq / fracture_den_sq)  
  
    return matrix_error, fracture_error

