from __future__ import annotations

import numpy as np
import porepy as pp

import helpers.mdamr_local  # adds external/mdamr-main/src to sys.path
from mdamr.non_matching.transfer_grid import TransferGrid


def _cell_nodes(sd: pp.Grid, expected: int) -> np.ndarray:
    """Return node indices for each cell of a simplex grid."""
    cn = sd.cell_nodes().tocsc()
    out: list[np.ndarray] = []
    for c in range(sd.num_cells):
        nodes = cn.indices[cn.indptr[c] : cn.indptr[c + 1]]
        if len(nodes) != expected:
            raise ValueError(
                f"Cell {c} has {len(nodes)} nodes, expected {expected}."
            )
        out.append(nodes)
    return np.asarray(out, dtype=int)


def _transfer_triangles(tr: pp.Grid) -> np.ndarray:
    """Return transfer-grid triangle connectivity, shape = (num_cells, 3)."""
    cn = tr.cell_nodes().tocsc()
    return cn.indices.reshape((3, tr.num_cells), order="F").T


def _triangle_area(xy: np.ndarray) -> float:
    """Area of a physical triangle with xy shape = (2, 3)."""
    p0 = xy[:, 0]
    p1 = xy[:, 1]
    p2 = xy[:, 2]
    return 0.5 * abs(np.linalg.det(np.column_stack((p1 - p0, p2 - p0))))


def _triangle_quadrature_degree4(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the degree-4 Dunavant quadrature rule on one physical triangle.

    Parameters
    ----------
    xy:
        Physical triangle vertices in the working 2D frame, shape = (2, 3).

    Returns
    -------
    qx:
        Quadrature points, shape = (2, 6).
    qw:
        Physical quadrature weights, shape = (6,).

    Notes
    -----
    The constants below are the Dunavant p=4, ng=6 rule. The weights used here
    sum to 0.5 on the reference triangle, so they are scaled by 2 * area to get
    physical weights.
    """
    a1 = 0.445948490915965
    b1 = 0.108103018168070
    w1 = 0.111690794839005

    a2 = 0.091576213509771
    b2 = 0.816847572980459
    w2 = 0.054975871827661

    bary = np.array(
        [
            [a1, a1, b1],
            [a1, b1, a1],
            [b1, a1, a1],
            [a2, a2, b2],
            [a2, b2, a2],
            [b2, a2, a2],
        ],
        dtype=float,
    )
    weights_ref = np.array([w1, w1, w1, w2, w2, w2], dtype=float)

    area = _triangle_area(xy)
    qx = xy @ bary.T
    qw = 2.0 * area * weights_ref
    return qx, qw


def _rotate_fem_nodes_to_transfer_frame(
    nodes_xy: np.ndarray,
    rotation_matrix: np.ndarray,
    dim_bool: np.ndarray,
) -> np.ndarray:
    """Rotate FEM nodes into the same 2D frame as TransferGrid.

    Parameters
    ----------
    nodes_xy:
        FEM node coordinates, shape = (num_nodes, 2).
    rotation_matrix:
        3x3 rotation matrix used internally by mdamr.TransferGrid.
    dim_bool:
        Active dimensions of the rotated grid.

    Returns
    -------
    nodes_rot:
        Rotated 2D node coordinates, shape = (num_nodes, 2).
    """
    nodes_xy = np.asarray(nodes_xy, dtype=float)
    if nodes_xy.ndim != 2 or nodes_xy.shape[1] != 2:
        raise ValueError("Expected nodes_xy with shape (num_nodes, 2).")

    nodes3 = np.zeros((3, nodes_xy.shape[0]), dtype=float)
    nodes3[:2, :] = nodes_xy.T
    rot = rotation_matrix @ nodes3
    active = rot[dim_bool, :]
    if active.shape[0] != 2:
        raise ValueError(
            f"Expected a 2D active frame after rotation, got {active.shape[0]}D."
        )
    return active.T


def _barycentric_coordinates(
    vertices: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates of points in a triangle.

    Parameters
    ----------
    vertices:
        Triangle vertices, shape = (2, 3), ordered as [v0, v1, v2].
    points:
        Points, shape = (2, num_points).

    Returns
    -------
    L:
        Barycentric coordinates, shape = (3, num_points), corresponding to
        [v0, v1, v2].
    """
    A = np.vstack((vertices, np.ones(3)))
    rhs = np.vstack((points, np.ones(points.shape[1])))
    return np.linalg.solve(A, rhs)


def _t6_shape_functions_from_barycentric(L: np.ndarray) -> np.ndarray:
    """Evaluate T6 shape functions from barycentric coordinates.

    The local T6 node order is assumed to be:
        [v0, m01, v1, m12, v2, m02]

    This matches the order used by the user's FEM mesh construction:
        v0, m01, v1, m12, v2, m02.

    Parameters
    ----------
    L:
        Barycentric coordinates, shape = (3, num_points), ordered as
        [L0, L1, L2] for vertices [v0, v1, v2].

    Returns
    -------
    N:
        T6 shape functions, shape = (6, num_points).
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
    """Convert flattened FEM displacement to shape = (2, num_nodes)."""
    U = np.asarray(U, dtype=float).ravel()
    if U.size % 2 != 0:
        raise ValueError("Expected flattened displacement vector with even length.")
    return np.vstack((U[0::2], U[1::2]))


def _eval_t6_displacement(
    qx: np.ndarray,
    element_nodes: np.ndarray,
    element_disp: np.ndarray,
) -> np.ndarray:
    """Evaluate a standard T6 displacement field at physical points.

    Parameters
    ----------
    qx:
        Evaluation points in the same 2D frame as element_nodes, shape = (2, n_q).
    element_nodes:
        Coordinates of the 6 T6 nodes, shape = (6, 2), ordered as
        [v0, m01, v1, m12, v2, m02].
    element_disp:
        Displacement values at the 6 T6 nodes, shape = (2, 6).

    Returns
    -------
    uq:
        Displacement at qx, shape = (2, n_q).

    Notes
    -----
    This evaluator assumes a standard straight-sided T6 element, where the physical
    triangle is represented by the corner vertices [v0, v1, v2]. This is appropriate
    for the non-QPE standard quadratic FEM. For QPE elements, a true isoparametric
    inverse mapping would be needed near tips.
    """
    if element_nodes.shape != (6, 2):
        raise ValueError(f"Expected element_nodes shape (6, 2), got {element_nodes.shape}.")
    if element_disp.shape != (2, 6):
        raise ValueError(f"Expected element_disp shape (2, 6), got {element_disp.shape}.")

    vertices = element_nodes[[0, 2, 4], :].T
    L = _barycentric_coordinates(vertices, qx)
    N = _t6_shape_functions_from_barycentric(L)
    return element_disp @ N


def _check_fem_model_ready(model: object) -> None:
    """Basic validation of the FEM model object expected by this module."""
    required = ["grid2d", "nodes_p6", "elements_t6", "U"]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(f"FEM model is missing required attributes: {missing}.")
    if model.nodes_p6 is None or model.elements_t6 is None or model.U is None:
        raise RuntimeError("FEM model has not been solved/setup yet. Call model_setup_and_sol().")
    if model.elements_t6.shape[0] != model.grid2d.num_cells:
        raise ValueError(
            "Expected elements_t6 rows to match grid2d cells. "
            f"Got {model.elements_t6.shape[0]} elements and {model.grid2d.num_cells} grid cells."
        )


def matrix_overlap_l2_components_t6_quad(
    coarse_model: object,
    fine_model: object,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Compute matrix-domain relative-L2 components for standard T6 FEM fields.

    It computes
        numerator_sq   = int_Omega |u_H^{T6} - u_h^{T6}|^2 dx
        denominator_sq = int_Omega |u_h^{T6}|^2 dx

    on the transfer grid generated by intersections of the coarse and fine
    PorePy matrix grids.

    Parameters
    ----------
    coarse_model, fine_model:
        Solved FEM models with attributes grid2d, nodes_p6, elements_t6 and U.
    tol:
        Tolerance passed to mdamr.TransferGrid.

    Returns
    -------
    numerator_sq, denominator_sq:
        Squared L2 numerator and denominator.
    """
    _check_fem_model_ready(coarse_model)
    _check_fem_model_ready(fine_model)

    sd_H = coarse_model.grid2d
    sd_h = fine_model.grid2d

    tg = TransferGrid(sd_H, sd_h, tol=tol, name="matrix_transfer_t6")
    src_rot = tg._get_rotated_grid(sd_H)
    tgt_rot = tg._get_rotated_grid(sd_h)

    # TransferGrid stores its transfer nodes in the common rotated frame. Rotate
    # the FEM nodes to the same frame before evaluating shape functions.
    nodes_H = _rotate_fem_nodes_to_transfer_frame(
        coarse_model.nodes_p6,
        src_rot.rotation_matrix,
        src_rot.dim_bool,
    )
    nodes_h = _rotate_fem_nodes_to_transfer_frame(
        fine_model.nodes_p6,
        tgt_rot.rotation_matrix,
        tgt_rot.dim_bool,
    )

    elems_H = np.asarray(coarse_model.elements_t6, dtype=int)
    elems_h = np.asarray(fine_model.elements_t6, dtype=int)

    UH_node = _fem_displacement_by_node(coarse_model.U)
    Uh_node = _fem_displacement_by_node(fine_model.U)

    tr = tg.transfer
    tr_cells = _transfer_triangles(tr)
    Xtr = tr.nodes[:2, :]

    t2s = tg.transfer_to_source.tocsr()
    t2t = tg.transfer_to_target.tocsr()

    numerator_sq = 0.0
    denominator_sq = 0.0

    for j, tr_nodes in enumerate(tr_cells):
        xy = Xtr[:, tr_nodes]
        qx, qw = _triangle_quadrature_degree4(xy)

        src_parent = t2s[j].indices
        tgt_parent = t2t[j].indices

        if src_parent.size != 1 or tgt_parent.size != 1:
            raise RuntimeError(
                f"Transfer triangle {j} has source parents {src_parent} "
                f"and target parents {tgt_parent}; expected one each."
            )

        KH = int(src_parent[0])
        kh = int(tgt_parent[0])

        eH = elems_H[KH]
        eh = elems_h[kh]

        uH_q = _eval_t6_displacement(
            qx=qx,
            element_nodes=nodes_H[eH],
            element_disp=UH_node[:, eH],
        )
        uh_q = _eval_t6_displacement(
            qx=qx,
            element_nodes=nodes_h[eh],
            element_disp=Uh_node[:, eh],
        )

        diff_q = uH_q - uh_q
        numerator_sq += float(np.sum(qw * np.sum(diff_q**2, axis=0)))
        denominator_sq += float(np.sum(qw * np.sum(uh_q**2, axis=0)))

    return numerator_sq, denominator_sq

# For displacement jump on 1D fracture
def _p2_shape_1d(xi: np.ndarray) -> np.ndarray:
    """Quadratic Lagrange shape functions on [-1, 1].

    Node order: left, middle, right.
    """
    return np.vstack([
        0.5 * xi * (xi - 1.0),
        1.0 - xi**2,
        0.5 * xi * (xi + 1.0),
    ])


# def _build_p2_segments_from_sorted_nodes(
#     s: np.ndarray,
#     values: np.ndarray,
# ) -> list[tuple[np.ndarray, np.ndarray]]:
#     """Build consecutive P2 segments from sorted fracture nodes.

#     Assumes node order along fracture is:
#         vertex, mid-node, vertex, mid-node, vertex, ...
#     """
#     segments = []

#     if len(s) < 3:
#         raise ValueError("Need at least three fracture nodes for P2 segments.")

#     for i in range(0, len(s) - 2, 2):
#         s_seg = s[i : i + 3]
#         v_seg = values[i : i + 3]

#         if not np.all(np.diff(s_seg) > -1e-12):
#             raise ValueError("Fracture coordinates are not sorted.")

#         segments.append((s_seg, v_seg))

#     return segments

def _fracture_p2_jump_segments_from_model(model: object):
    theta = model.theta_rad
    t = np.array([np.sin(theta), np.cos(theta)])
    n = np.array([-np.cos(theta), np.sin(theta)])

    ux = model.U[0::2]
    uy = model.U[1::2]

    upper_edges = model.upper_frac_edge_dofs
    lower_edges = model.lower_frac_edge_dofs

    segs_n = []
    segs_t = []

    for upper, lower in zip(upper_edges, lower_edges):
        upper = np.asarray(upper, dtype=int)
        lower = np.asarray(lower, dtype=int)

        pts = model.nodes_p6[upper, :2]
        s = pts @ t

        idx = np.argsort(s)
        s = s[idx]
        upper = upper[idx]
        lower = lower[idx]

        u_upper = np.column_stack((ux[upper], uy[upper]))
        u_lower = np.column_stack((ux[lower], uy[lower]))

        jump = u_upper - u_lower

        segs_n.append((s, jump @ n))
        segs_t.append((s, jump @ t))

    segs_n.sort(key=lambda seg: seg[0][0])
    segs_t.sort(key=lambda seg: seg[0][0])

    return segs_n, segs_t

def _eval_p2_segment_1d(
    s_query: np.ndarray,
    s_seg: np.ndarray,
    v_seg: np.ndarray,
) -> np.ndarray:
    """Evaluate a scalar P2 field on one physical 1D segment."""
    s0 = float(s_seg[0])
    s2 = float(s_seg[2])

    if abs(s2 - s0) < 1e-14:
        raise RuntimeError("Degenerate P2 fracture segment.")

    xi = 2.0 * (s_query - s0) / (s2 - s0) - 1.0
    N = _p2_shape_1d(xi)

    return v_seg @ N


def _find_parent_segment(
    segments: list[tuple[np.ndarray, np.ndarray]],
    s_mid: float,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the P2 segment containing s_mid."""
    for s_seg, v_seg in segments:
        s_left = min(float(s_seg[0]), float(s_seg[2]))
        s_right = max(float(s_seg[0]), float(s_seg[2]))

        if s_left - tol <= s_mid <= s_right + tol:
            return s_seg, v_seg

    raise RuntimeError(f"No parent P2 segment found for s = {s_mid}.")

def fracture_overlap_l2_components_p2_quad(
    coarse_model: object,
    fine_model: object,
    tol: float = 1e-8,
    num_quad: int = 3,
) -> tuple[float, float, float, float]:

    segs_H_n, segs_H_t = _fracture_p2_jump_segments_from_model(coarse_model)
    segs_h_n, segs_h_t = _fracture_p2_jump_segments_from_model(fine_model)

    H_endpoints = np.array([seg[0][0] for seg in segs_H_n] + [segs_H_n[-1][0][2]])
    h_endpoints = np.array([seg[0][0] for seg in segs_h_n] + [segs_h_n[-1][0][2]])

    s_min = max(np.min(H_endpoints), np.min(h_endpoints))
    s_max = min(np.max(H_endpoints), np.max(h_endpoints))

    overlap_nodes = np.unique(
        np.round(
            np.concatenate([
                H_endpoints[(H_endpoints >= s_min - tol) & (H_endpoints <= s_max + tol)],
                h_endpoints[(h_endpoints >= s_min - tol) & (h_endpoints <= s_max + tol)],
            ]),
            decimals=14,
        )
    )

    xi, wi = np.polynomial.legendre.leggauss(num_quad)

    normal_num_sq = 0.0
    normal_den_sq = 0.0
    tangential_num_sq = 0.0
    tangential_den_sq = 0.0

    for s0, s1 in zip(overlap_nodes[:-1], overlap_nodes[1:]):
        s0 = float(s0)
        s1 = float(s1)

        if abs(s1 - s0) < tol:
            continue

        smid = 0.5 * (s0 + s1)

        s_seg_H_n, v_seg_H_n = _find_parent_segment(segs_H_n, smid, tol=tol)
        s_seg_h_n, v_seg_h_n = _find_parent_segment(segs_h_n, smid, tol=tol)

        s_seg_H_t, v_seg_H_t = _find_parent_segment(segs_H_t, smid, tol=tol)
        s_seg_h_t, v_seg_h_t = _find_parent_segment(segs_h_t, smid, tol=tol)

        mid = 0.5 * (s0 + s1)
        half = 0.5 * (s1 - s0)

        sq = mid + half * xi
        wq = abs(half) * wi

        jumpH_n_q = _eval_p2_segment_1d(sq, s_seg_H_n, v_seg_H_n)
        jumph_n_q = _eval_p2_segment_1d(sq, s_seg_h_n, v_seg_h_n)

        jumpH_t_q = _eval_p2_segment_1d(sq, s_seg_H_t, v_seg_H_t)
        jumph_t_q = _eval_p2_segment_1d(sq, s_seg_h_t, v_seg_h_t)

        diff_n_q = jumpH_n_q - jumph_n_q
        diff_t_q = jumpH_t_q - jumph_t_q

        normal_num_sq += float(np.sum(wq * diff_n_q**2))
        normal_den_sq += float(np.sum(wq * jumph_n_q**2))

        tangential_num_sq += float(np.sum(wq * diff_t_q**2))
        tangential_den_sq += float(np.sum(wq * jumph_t_q**2))

    return (
        normal_num_sq,
        normal_den_sq,
        tangential_num_sq,
        tangential_den_sq,
    )

