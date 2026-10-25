"""
Convergence functionlities of P1 fields for the MPSA model containing a fracture
in 2D.

The reference solutions on 2D grids and across the 1D fracture are determined by the
functions implenmented for each benchmark.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import porepy as pp

from global_conv import QuadratureIntegration

def _cell_nodes(sd: pp.Grid) -> np.ndarray:
    """Get node indices for each cell. 
    
    Only simplex grids are supported: 
        - 2D triangles  -> 3 nodes: [node_i, node_j, node_k].
        - 1D segments   -> 2 nodes: [node_i, node_j]. 

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

    Cell-center variables should be vector. 
    Particularly, displacement variables are vector.

    Returns: 
        values_cc: shape (n_components, num_nodes) for vector.
    """
    n_comp = values_cc.shape[0]
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

    return u_node


def _p1_coefficients_from_nodal_values_2d(
    sd: pp.Grid,
    nodal_values: np.ndarray,
    node_xy: np.ndarray,
) -> np.ndarray:
    """Compute cell-wise coefficients as a P1 continuous field in 2D.

    Parameters: 
        sd: 2D grid. 
        nodal_values: vector, shape = (n_components, num_nodes).
        node_xy: nodal coordinates, shape = (2, num_nodes).

    Returns: 
        coeff: shape = (n_components, num_cells, 3).
            For each component and cell, 3 coefficients [a, b, c] define the field. 
            coeff[comp, cell] = [a, b, c] represents u(x,y) = a*x + b*y + c.
    """
    n_comp = nodal_values.shape[0]
    cells = _cell_nodes(sd)

    coeff = np.zeros((n_comp, sd.num_cells, 3))

    for i, nodes in enumerate(cells):
        xy = node_xy[:, nodes] 
        # Construct the matrix for the three vertices of the triangle. 
        # A = [[x0, y0, 1],  -> node 0
        #      [x1, y1, 1],  -> node 1
        #      [x2, y2, 1]]. -> node 2
        A = np.column_stack((xy[0], xy[1], np.ones(3)))  

        for j in range(n_comp):
            rhs = nodal_values[j, nodes]
            coeff[j, i, :] = np.linalg.solve(A, rhs)

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


def matrix_error_integration_p1_quad(
    sd: pp.Grid,
    u_num: np.ndarray,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    weight: str = "volume",
    num_quad: int = 7,
) -> tuple[float, float]:
    """Compute squared L2 error intergrals for the P1 field on the 2D grid using the quadrature
    rule, based on the given reference solution.

    It computes: 
        numerator_sq   = ∫_Omega ||u_num - u_ref||^2 dx
        denominator_sq = ∫_Omega ||u_ref||^2 dx
    where the reconstructed numerical field and the reference field are on the same 2D grid.

    Parameters:
        sd: 2D grid.
        u_num: numerical solution on the grid, shape = (n_components, num_cells).
        reference_solution: reference solution function by accepting coordinates with 
            shape = (2, n_points) and returning displacement components with
            shape = (n_components, n_points).
        weight: weight used for the nodal reconstruction, either "volume" or "uniform".
        num_quad: parameter of the Gaussian quadrature rule.
    
    Returns:
        numerator_sq: squared L2 error integral over the matrix domain.
        denominator_sq: squared L2 norm of the reference solution over the matrix domain.
    """
    # Reconstruct nodal values from cell-center values on the original grids.
    u_node = _nodal_average_from_cell_values(sd, u_num, weight=weight)

    node_xy = sd.nodes[:2, :]

    # Build the P1 field.
    coeff_p1 = _p1_coefficients_from_nodal_values_2d(
        sd, u_node, node_xy
    )

    cells = _cell_nodes(sd)

    numerator_sq = 0.0
    denominator_sq = 0.0

    for cell_id, nodes in enumerate(cells):
        xy = node_xy[:, nodes]
        
        quad_x, quad_w = (
            QuadratureIntegration.triangle_quadrature_gaussian(
                xy,
                num_quad,
            )
        )

        u_num_q = _eval_p1_2d(
            coeff_p1[:, cell_id, :],
            quad_x,
        )
        u_ref_q = reference_solution(quad_x)
        diff_q = u_num_q - u_ref_q

        numerator_sq += float(np.sum(quad_w * np.sum(diff_q**2, axis=0)))
        denominator_sq += float(np.sum(quad_w * np.sum(u_ref_q**2, axis=0)))

    return numerator_sq, denominator_sq

def _p1_coefficients_from_nodal_values_1d(
    sd: pp.Grid,
    nodal_values: np.ndarray,
    node_s: np.ndarray,
) -> np.ndarray:
    """Compute cell-wise coefficients as a P1 continuous field in a 1D line.

    Parameters: 
        sd: 1D grid.
        nodal_values: vector, shape = (n_components, num_nodes).
        node_s: signed physical length coordinates measured from the fracture
            center, shape = (num_nodes,).

    Returns: 
        coeff: shape = (n_components, num_cells, 2)
            For each component and cell, two coefficients [a, b] define the field. 
            coeff[comp, cell] = [a, b] represents u(s) = a*s + b.
    """
    n_comp = nodal_values.shape[0]
    cells = _cell_nodes(sd)

    coeff = np.zeros((n_comp, sd.num_cells, 2))

    for i, (n0, n1) in enumerate(cells):
        s0 = float(node_s[n0])
        s1 = float(node_s[n1])

        for j in range(n_comp):
            v0 = float(nodal_values[j, n0])
            v1 = float(nodal_values[j, n1])

            a = (v1 - v0) / (s1 - s0)
            b = v0 - a * s0

            coeff[j, i, :] = [a, b]

    return coeff


def _eval_p1_1d(coeff: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the P1 field at given points in 1D.
    
    Parameters: 
        coeff: shape = (n_components, 2)
        s: shape = (n_points,)

    Returns:
        shape = (n_components, n_points)
    """
    return coeff[:, 0:1] * s[None, :] + coeff[:, 1:2]


def fracture_error_integration_p1_quad(
    sd: pp.Grid,
    u_num: np.ndarray,
    reference_solution: Callable[[np.ndarray], np.ndarray],
    fracture_center: np.ndarray,
    tangent: np.ndarray,
    weight: str = "volume",
    num_quad: int = 7,
) -> tuple[float, float]:
    """Compute squared L2 error intergrals for the P1 field on 1D grids using the quadrature
    rule, based on the given reference solution.

    It computes: 
        numerator_sq   = ∫_Gamma ||u_num - u_ref||^2 ds
        denominator_sq = ∫_Gamma ||u_ref||^2 ds

    The reconstructed numerical field and the reference field are on the same 1D grid.

    The high-order quadrature is used to ensure the accuracy of the integration, since here
    a important usage of this method is to compare the reconstructed numerical field with the
    analytical solution of displacement jump.

    Parameters:
        sd: A straight 1D grid.
        u_num: numerical solution on the grid, shape = (n_components, num_cells).
        reference_solution: reference solution function by accepting coordinates with 
            shape = (n_points,) and returning displacement (or jump) components with
            shape = (n_components, n_points).
        fracture_center: 2D coordinates of the fracture center, shape = (2,).
        tangent: Unit tangent defining the positive direction of the fracture, shape = (2,).
        weight: weight used for the nodal reconstruction, either "volume" or "uniform".
        num_quad: parameter of the Gaussian quadrature rule.

    Returns:
        numerator_sq: squared L2 error integral over the 1D domain.
        denominator_sq: squared L2 norm of the reference solution over the 1D domain.   
    """
    # Reconstruct nodal values from cell-center values on the original grids.
    u_node = _nodal_average_from_cell_values(sd, u_num, weight=weight)

    node_xy = sd.nodes[:2, :]
    node_s = tangent @ (node_xy - fracture_center[:, None])

    # Compute coefficients of the P1 field along the fracture.
    coeff_p1 = _p1_coefficients_from_nodal_values_1d(sd, u_node, node_s)

    cells = _cell_nodes(sd)

    numerator_sq = 0.0
    denominator_sq = 0.0
    for cell_id, (n0, n1) in enumerate(cells):
        s0 = node_s[n0]
        s1 = node_s[n1]

        quad_s, quad_w = (
            QuadratureIntegration.segment_quadrature_gauss(
                s0,
                s1,
                num_quad=num_quad,
            )
        )

        u_num_q = _eval_p1_1d(coeff_p1[:, cell_id, :], quad_s)
        u_ref_q = reference_solution(quad_s)

        diff_q = u_num_q - u_ref_q

        numerator_sq += float(np.sum(quad_w * np.sum(diff_q**2, axis=0)))
        denominator_sq += float(np.sum(quad_w * np.sum(u_ref_q**2, axis=0)))

    return numerator_sq, denominator_sq


# Extract cell-centered displacement on 2D matrix grid. 
def extract_matrix_displacement_cc(
        model: object
    ) -> np.ndarray:
    """Compute cell-centered displacement on the 2D matrix grid.

    Parameters:
        model: A solved PorePy model.

    Returns: 
        cell-centered displacement (shape=(2, num_cells)).
    """
    sd = model.mdg.subdomains(dim=2)[0]
    u = model.displacement([sd]).value(model.equation_system)
    return u.reshape((model.nd, sd.num_cells), order="F")


# Extract cell-centered displacement jump on 1D fracture. 
def extract_fracture_displacement_jump_cc(
        model: object
    ) -> np.ndarray:
    """Compute cell-centered displacement jump on the 1D fracture cells.

    Parameters:
        model: A solved PorePy model.

    Returns:
        cell-centered displacement (shape=(2, num_cells)).
    """
    sd = model.mdg.subdomains(dim=1)[0]
    u_jump = model.displacement_jump([sd]).value(
        model.equation_system
    )
    return u_jump.reshape((model.nd, sd.num_cells), order="F")


def relative_ref_error_p1_mpsa(
    model: object,
    reference_solution_2d: Callable[[np.ndarray], np.ndarray],
    reference_solution_1d: Callable[[np.ndarray], np.ndarray],
    fracture_center: np.ndarray,
    tangent: np.ndarray,
    averaging_weight: str = "volume",
    num_quad_matrix: int = 7,
    num_quad_fracture: int = 7,
) -> tuple[float, float]:
    """Compute the relative L2 errors of the reconstructed MPSA P1 fields
    with respect to the reference solutions.

    Parameters: 
        model: A solved PorePy model.
        reference_solution_2d: reference solution function by accepting coordinates with 
            shape = (2, n_points) and returning displacement components with
            shape = (n_components, n_points).
        reference_solution_1d: reference solution function by accepting coordinates with 
            shape = (n_points,) and returning displacement (or jump) components with
            shape = (n_components, n_points).
        fracture_center: 2D coordinates of the fracture center, shape = (2,).
        tangent: Unit tangent defining the positive direction of the fracture, shape = (2,).
        averaging_weight: weight used for the nodal reconstruction, either "volume" or "uniform".
        num_quad_matrix: parameter of the Gaussian quadrature rule for the 2D matrix domain
        num_quad_fracture: parameter of the Gaussian quadrature rule for the 1D fracture domain.

    Returns:
        matrix_error: relative L2 error of the reconstructed P1 field on the 2D matrix domain.
        fracture_error: relative L2 error of the reconstructed P1 field on the 1D fracture domain.
    """
    sd_matrix = model.mdg.subdomains(dim=2)[0]

    # Cell-centered displacement on the 2D matrix grid.
    u_num = extract_matrix_displacement_cc(model)

    matrix_num_sq, matrix_den_sq = (
        matrix_error_integration_p1_quad(
            sd=sd_matrix,
            u_num=u_num,
            reference_solution=reference_solution_2d,
            weight=averaging_weight,
            num_quad=num_quad_matrix,
        )
    )

    sd_fracture = model.mdg.subdomains(dim=1)[0]
    u_jump_num = extract_fracture_displacement_jump_cc(model)

    fracture_num_sq, fracture_den_sq = (
        fracture_error_integration_p1_quad(
            sd=sd_fracture,
            u_num=u_jump_num,
            reference_solution=reference_solution_1d,
            fracture_center=fracture_center,
            tangent=tangent,
            weight=averaging_weight,
            num_quad=num_quad_fracture,
        )
    )

    matrix_error = np.sqrt(matrix_num_sq / matrix_den_sq)
    fracture_error = np.sqrt(fracture_num_sq / fracture_den_sq)

    return matrix_error, fracture_error
