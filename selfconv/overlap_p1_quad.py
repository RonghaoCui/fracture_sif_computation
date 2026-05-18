from __future__ import annotations

import numpy as np
import porepy as pp

import helpers.mdamr_local  # adds external/mdamr-main/src to sys.path

# import quadpy
# from mdamr.utils.quadrature import get_quadpy_elements
from mdamr.non_matching.transfer_grid import TransferGrid, TransferLine


def _cell_nodes(sd: pp.Grid) -> np.ndarray:
    """Get node indices for each cell. 
    
    Only simplex grids are supported: 
        - 2D triangles  -> 3 nodes
        - 1D segments   -> 2 nodes. 

    Parameters:
        sd: grid        

    Returns:
        node indices: shape = (num_cells, dim + 1)
    """
    cn = sd.cell_nodes().tocsc()
    return cn.indices.reshape((sd.dim + 1, sd.num_cells), order="F").T 


def _nodal_average_from_cell_values(
    sd: pp.Grid,
    values_cc: np.ndarray,
    weight: str = "volume",
) -> np.ndarray:
    """Reconstruction of nodal values from cell-center values. 

    The weight can be either "volume" or "uniform". 
    The former gives a volume-weighted average, while the latter gives a simple average.

    Cell-center variables can be scalar or vector. 
    Particularly, displacement variables are vector with n_components = grid dimension.

    Returns: 
        values_cc: shape (n_components, num_cells)
            or shape (num_cells,) for scalar.
    """
    # Check if values_cc is scalar (shape (num_cells,)) or vector (shape (n_components, num_cells)).
    scalar = values_cc.ndim == 1
    if scalar:
        values_cc = values_cc[None, :]

    n_comp, n_cells = values_cc.shape

    u_node = np.zeros((n_comp, sd.num_nodes))
    w_node = np.zeros(sd.num_nodes)

    cells = _cell_nodes(sd)

    for c, nodes in enumerate(cells):
        if weight == "volume":
            w = float(sd.cell_volumes[c])
        elif weight == "uniform":
            w = 1.0
        else:
            raise ValueError("weight must be 'volume' or 'uniform'.")

        u_node[:, nodes] += w * values_cc[:, c, None]
        w_node[nodes] += w

    u_node /= w_node[None, :]

    return u_node[0] if scalar else u_node


def _p1_coefficients_from_nodal_values_2d(
    sd: pp.Grid,
    nodal_values: np.ndarray,
    node_xy: np.ndarray,
) -> np.ndarray:
    """Compute cell-wise coefficients as a P1 continuous field in 2D.

    Parameters: 
        sd: 2D grid. 
        nodal_values: shape = (n_components, num_nodes) or (num_nodes,).
        node_xy: shape = (2, num_nodes), in the same coordinate frame used 
            by TransferGrid.

    Returns: 
        coeff: shape = (n_components, num_cells, 3).
            For each component and cell, 3 coefficients [a, b, c] define the field. 
            coeff[comp, cell] = [a, b, c] represents u(x,y) = a*x + b*y + c.
    """
    nodal_values = np.asarray(nodal_values)
    if nodal_values.ndim == 1:
        nodal_values = nodal_values[None, :]

    n_comp = nodal_values.shape[0]
    cells = _cell_nodes(sd)

    coeff = np.zeros((n_comp, sd.num_cells, 3))

    for c, nodes in enumerate(cells):
        xy = node_xy[:, nodes]  # (2, 3)
        A = np.column_stack((xy[0], xy[1], np.ones(3)))  # (3, 3)

        for m in range(n_comp):
            rhs = nodal_values[m, nodes]
            coeff[m, c, :] = np.linalg.solve(A, rhs)

    return coeff


def _eval_p1_2d(coeff: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    Compute the P1 field at given points in 2D.

    Parameters:
        coeff: shape = (n_components, 3)
        xy: shape = (2, n_points)

    Returns: 
        shape = (n_components, n_points)
    """
    return coeff[:, 0:1] * xy[0][None, :] + coeff[:, 1:2] * xy[1][None, :] + coeff[:, 2:3]


def _triangle_area(xy: np.ndarray) -> float:
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


def _transfer_triangles(tr: pp.Grid) -> np.ndarray:
    cn = tr.cell_nodes().tocsc()
    return cn.indices.reshape((3, tr.num_cells), order="F").T


def _triangle_quadrature_degree4(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the degree-4 quadrature rule on one physical triangle.

    Proposed by Dunavant (1985), this method offers 6-point quadrature consisting 
    of two groups. Each group consists of 3 quadrature points, resulting in 
    a total of 6 quadrature points. With the same weights, the quadrature points 
    in each group have a format of barycentric coordinates, (a, a, b), 
    where a + a + b = 1. 

    The reference can be found in: 
        - Dunavant, D. (1985). International Journal for Numerical Methods in 
        Engineering, 21(6), 1129-1148.

    Parameters: 
        xy: triangle vertices, shape = (2, 3).

    Returns: 
        qx: quadrature points, shape = (2, num_quad).
        qw: weights, shape = (num_quad,).
    """
    # Dunavant 6-point rule, degree 4.
    a1 = 0.445948490915965
    b1 = 0.108103018168070
    w1 = 0.111690794839005

    a2 = 0.091576213509771
    b2 = 0.816847572980459
    w2 = 0.054975871827661

    bary = np.array([
        [a1, a1, b1],
        [a1, b1, a1],
        [b1, a1, a1],
        [a2, a2, b2],
        [a2, b2, a2],
        [b2, a2, a2],
    ])

    weights_ref = np.array([w1, w1, w1, w2, w2, w2])

    area = _triangle_area(xy)
    qx = xy @ bary.T

    # Dunavant weights sum to 0.5 on the reference triangle.
    # Therefore multiply by 2 * physical area.
    qw = 2.0 * area * weights_ref

    return qx, qw


def matrix_overlap_l2_components_p1_quad(
    sd_H: pp.Grid,
    sd_h: pp.Grid,
    uH_cc: np.ndarray,
    uh_cc: np.ndarray,
    weight: str = "volume",
    tol: float = 1e-8,
) -> tuple[float, float]:
    """
    Access Jhabriel's grid transfer generation for 2D matrix. 

    It computes: 
        numerator_sq   = ∫_Omega |u_H^P1 - u_h^P1|^2 dx
        denominator_sq = ∫_Omega |u_h^P1|^2 dx
    
    on the transfer grid generated by intersections of coarse and fine cells.

    """
    tg = TransferGrid(sd_H, sd_h, tol=tol, name="matrix_transfer")

    src_rot = tg._get_rotated_grid(sd_H)
    tgt_rot = tg._get_rotated_grid(sd_h)

    uH_node = _nodal_average_from_cell_values(sd_H, uH_cc, weight=weight)
    uh_node = _nodal_average_from_cell_values(sd_h, uh_cc, weight=weight)

    CH = _p1_coefficients_from_nodal_values_2d(
        sd_H, uH_node, src_rot.nodes[:2, :]
    )
    Ch = _p1_coefficients_from_nodal_values_2d(
        sd_h, uh_node, tgt_rot.nodes[:2, :]
    )

    tr = tg.transfer
    tr_cells = _transfer_triangles(tr)
    Xtr = tr.nodes[:2, :]

    t2s = tg.transfer_to_source.tocsr()
    t2t = tg.transfer_to_target.tocsr()

    numerator_sq = 0.0
    denominator_sq = 0.0

    for j, nodes in enumerate(tr_cells):
        xy = Xtr[:, nodes]

        src_parent = t2s[j].indices
        tgt_parent = t2t[j].indices

        KH = int(src_parent[0])
        kh = int(tgt_parent[0])

        qx, qw = _triangle_quadrature_degree4(xy)

        uH_q = _eval_p1_2d(CH[:, KH, :], qx)
        uh_q = _eval_p1_2d(Ch[:, kh, :], qx)

        diff_q = uH_q - uh_q

        numerator_sq += float(np.sum(qw * np.sum(diff_q**2, axis=0)))
        denominator_sq += float(np.sum(qw * np.sum(uh_q**2, axis=0)))

    return numerator_sq, denominator_sq


def _p1_coefficients_from_nodal_values_1d(
    sd: pp.Grid,
    nodal_values: np.ndarray,
    node_s: np.ndarray,
) -> np.ndarray:
    """Compute cell-wise coefficients as a P1 continuous field in a 1D line.

    Parameters: 
        sd: 1D grid.
        nodal_values: shape = (n_components, num_nodes)  or (num_nodes,).
        node_s: shape = (1, num_nodes), in the same coordinate frame used 
            by TransferGrid.

    Returns: 
        coeff: shape = (n_components, num_cells, 2)
            For each component and cell, 2 coefficients [a, b] define the field. 
            coeff[comp, cell] = [a, b] represents u(s) = a*s + b.
    """
    nodal_values = np.asarray(nodal_values)

    if nodal_values.ndim == 1:
        nodal_values = nodal_values[None, :]

    n_comp = nodal_values.shape[0]
    cells = _cell_nodes(sd)

    coeff = np.zeros((n_comp, sd.num_cells, 2))

    for c, (n0, n1) in enumerate(cells):
        s0 = float(node_s[n0])
        s1 = float(node_s[n1])

        if abs(s1 - s0) < 1e-14:
            raise RuntimeError(f"Degenerate 1D cell {c}.")

        for m in range(n_comp):
            v0 = float(nodal_values[m, n0])
            v1 = float(nodal_values[m, n1])

            a = (v1 - v0) / (s1 - s0)
            b = v0 - a * s0

            coeff[m, c, :] = [a, b]

    return coeff


def _eval_p1_1d(coeff: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the P1 field at points in 1D.
    
    Parameters: 
        coeff: shape = (n_components, 2)
        s: shape = (n_points,)

    Returns:
        shape = (n_components, n_points)
    """
    return coeff[:, 0:1] * s[None, :] + coeff[:, 1:2]


def _segment_quadrature_gauss(
    s0: float,
    s1: float,
    num_quad: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss-Legendre quadrature rule on one 1D segment [s0, s1].
    Here, we use the standard Gauss-Legendre quadrature, which is exact 
    for polynomials of degree up to 2*num_quad - 1. Therefore, in order to cover
    the P1 and P2 fields (whose squared errors have degrees of 2 and 4, repectively), 
    num_quad = 3 is used.

    Returns: 
        qs: quadrature points, shape = (num_quad,).
        qw: weights, shape = (num_quad,).
    """
    xi, wi = np.polynomial.legendre.leggauss(num_quad)

    mid = 0.5 * (s0 + s1)
    half = 0.5 * (s1 - s0)

    qs = mid + half * xi
    qw = abs(half) * wi

    return qs, qw


def fracture_overlap_l2_components_p1_quad(
    sd_H: pp.Grid,
    sd_h: pp.Grid,
    wH_cc: np.ndarray,
    wh_cc: np.ndarray,
    weight: str = "volume",
    tol: float = 1e-8,
    num_quad: int = 3,
) -> tuple[float, float]:
    """
    Jhabriel's grid transfer generation for 1D fracture.
    
    Here, the variable is displacement jump vector. 
    
    Using num_quad = 3 is enough to cover the squared error of P1 and P2 fields 
    (degrees 2 and 4, respectively).

    It computes: 
        numerator_sq   = ∫_Gamma |w_H^P1 - w_h^P1|^2 ds
        denominator_sq = ∫_Gamma |w_h^P1|^2 ds
    """
    # if sd_H.dim != 1 or sd_h.dim != 1:
    #     raise ValueError("fracture_overlap_l2_components_p1 expects 1D grids.")

    tl = TransferLine(sd_H, sd_h, tol=tol, name="fracture_transfer")

    wH_cc = np.asarray(wH_cc)
    wh_cc = np.asarray(wh_cc)

    if wH_cc.ndim == 1:
        wH_cc = wH_cc[None, :]
    if wh_cc.ndim == 1:
        wh_cc = wh_cc[None, :]

    # if wH_cc.shape[1] != sd_H.num_cells:
    #     raise ValueError(f"wH_cc shape {wH_cc.shape} incompatible with {sd_H.num_cells} cells.")
    # if wh_cc.shape[1] != sd_h.num_cells:
    #     raise ValueError(f"wh_cc shape {wh_cc.shape} incompatible with {sd_h.num_cells} cells.")

    wH_node = _nodal_average_from_cell_values(sd_H, wH_cc, weight=weight)
    wh_node = _nodal_average_from_cell_values(sd_h, wh_cc, weight=weight)

    sH = tl._x_in_common_frame(sd_H, use_rotation=tl._rotate_source)
    sh = tl._x_in_common_frame(sd_h, use_rotation=tl._rotate_target)

    CH = _p1_coefficients_from_nodal_values_1d(sd_H, wH_node, sH)
    Ch = _p1_coefficients_from_nodal_values_1d(sd_h, wh_node, sh)

    xtr = tl.transfer.nodes[0, :]
    t2s = tl.transfer_to_source.tocsr()
    t2t = tl.transfer_to_target.tocsr()

    numerator_sq = 0.0
    denominator_sq = 0.0

    for j in range(tl.transfer.num_cells):
        s0 = float(xtr[j])
        s1 = float(xtr[j + 1])

        src_parent = t2s[j].indices
        tgt_parent = t2t[j].indices

        KH = int(src_parent[0])
        kh = int(tgt_parent[0])

        qs, qw = _segment_quadrature_gauss(s0, s1, num_quad=num_quad)

        wH_q = _eval_p1_1d(CH[:, KH, :], qs)
        wh_q = _eval_p1_1d(Ch[:, kh, :], qs)

        diff_q = wH_q - wh_q

        numerator_sq += float(np.sum(qw * np.sum(diff_q**2, axis=0)))
        denominator_sq += float(np.sum(qw * np.sum(wh_q**2, axis=0)))

    return numerator_sq, denominator_sq

def interface_overlap_l2_components_p1_quad(
    sd_H: pp.Grid,
    sd_h: pp.Grid,
    mH_cc: np.ndarray,
    mh_cc: np.ndarray,
    weight: str = "volume",
    tol: float = 1e-8,
    num_quad: int = 3,
) -> tuple[float, float]:
    """
    Access Jhabriel's grid transfer generation for two-sided mortar/interface (1D).
    
    Using num_quad = 3 is enough to cover the squared error of P1 and P2 fields 
    (degrees 2 and 4, respectively).

    Parameters: 
        sd_H, sd_h: 1D fracture grids, used as geometry for each mortar side.
        mH_cc, mh_cc: Interface displacement on mortar cells.
            shape = (nd, 2 * num_fracture_cells)

        The two mortar sides are assumed ordered as:
            first num_fracture_cells cells  -> side 0
            second num_fracture_cells cells -> side 1

    Computes:
        sum_side ∫_Gamma ||u_H^side - u_h^side||^2 ds
    """
    # mH_cc = np.asarray(mH_cc)
    # mh_cc = np.asarray(mh_cc)

    # BUG: need to check whether two sides should be split.
    H_side_0 = mH_cc[:, :sd_H.num_cells]
    H_side_1 = mH_cc[:, sd_H.num_cells:]

    h_side_0 = mh_cc[:, :sd_h.num_cells]
    h_side_1 = mh_cc[:, sd_h.num_cells:]

    num0, den0 = fracture_overlap_l2_components_p1_quad(
        sd_H=sd_H,
        sd_h=sd_h,
        wH_cc=H_side_0,
        wh_cc=h_side_0,
        weight=weight,
        tol=tol,
        num_quad=num_quad,
    )

    num1, den1 = fracture_overlap_l2_components_p1_quad(
        sd_H=sd_H,
        sd_h=sd_h,
        wH_cc=H_side_1,
        wh_cc=h_side_1,
        weight=weight,
        tol=tol,
        num_quad=num_quad,
    )

    return num0 + num1, den0 + den1
