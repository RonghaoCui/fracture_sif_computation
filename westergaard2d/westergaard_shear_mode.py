"""
Model for Westergaard's problem in the shear mode 
using Westergaard's analytical solution for tangential displacement jump.

References:

    - [1] Starfield, C.: Boundary Element Methods in Solid Mechanics (1983)
    - [2] Gdoutos, E.E.: Fracture mechanics: an introduction (2005)

"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.geometry.distances import point_pointset
from tests.functional.setups.manu_sneddon_2d import (
    ManuSneddonGeometry2d,
    ManuSneddonBoundaryConditions,
    SneddonExactSolution2d,
    ManuSneddonDataSaving,
)

class WestergaardShearModeExactSolution2d(SneddonExactSolution2d):
    """
    Class representing Westergaard's analytical solution in the shear mode 
    for tangential displacement jump in 2D.

    This class extends :class:`SneddonExactSolution2d`. 
        - Overridden methods:
                :meth:`exact_sol_global`
                :meth:`displacement_bc_values_from_bem`. 
        - Reused methods from parent: 
                :meth:`get_bem_centers`
                :meth:`tranform_boundary_face_coordinates_bem`

    Attributes:
        tau_inf: Shear stress at infinity.
    """
    def __init__(self, model) -> None:
        """
        Initialize attributes in parent :class:`SneddonExactSolution2d` 
        and store the shear stress at infinity ``tau_inf``

        Parameters: 
            model: Model instance with problem parameters. 
        """
        super().__init__(model)
        self.tau_inf = float(model.params.get("tau_inf"))

    def exact_sol_global(self, sd: pp.Grid) -> np.ndarray:
        """
        Compute the displacement values on boundary faces by superposing the
        fracture-induced perturbation (BEM with tangential slip source) and the
        centered far-field shear background.

        Parameters: 
            sd: Subdomain for analytical displacement values on boundary faces.

        Returns: 
            Displacement values on boundary faces with shape ``(dim, num_faces)``.

        Note: 
            The far-field shear strain is centered at the domain midpoint in order to 
            remove rigid translation. 
        """
        h = 2 * self.a / self.n
        box_faces = sd.get_boundary_faces()

        center = np.array([self.length / 2, self.length / 2, 0.0])
        bem_centers = self.get_bem_centers(h, center)
        eta = point_pointset(bem_centers, center)

        v_a = -self.analytical_tangential_slip(eta)
        u_frac = self.displacement_bc_values_from_bem(sd, h / 2, box_faces, bem_centers, v_a)

        # Rotate far-field shear displacements to global coordinates
        t = np.array([np.sin(self.theta_rad), np.cos(self.theta_rad)])
        n = np.array([-np.cos(self.theta_rad), np.sin(self.theta_rad)])

        x = sd.face_centers[0, box_faces]
        y = sd.face_centers[1, box_faces]

        dx = x - center[0]
        dy = y - center[1]
        r_t = dx * t[0] + dy * t[1]   # Local tangential coordinate
        r_n = dx * n[0] + dy * n[1]   # Local normal coordinate

        eps_coeff = self.tau_inf / (2.0 * self.shear_modulus)
        u_inf = np.zeros((sd.dim, sd.num_faces))

        # cx, cy = center[:2]
        u_inf[0, box_faces] = eps_coeff * (r_n * t[0] + r_t * n[0])
        u_inf[1, box_faces] = eps_coeff * (r_n * t[1] + r_t * n[1])

        return u_inf + u_frac
    
    def exact_sol_tangential_fracture(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """
        Compute Westergaard's analytical solution in the shear mode for the question.

        Parameters:
            mdg: Mixed-dimensional domain of the model.

        Return:
            A numpy array with the analytical slips at fracture face centers. 
        """
        ambient_dim = mdg.dim_max()
        g_1 = mdg.subdomains(dim=ambient_dim - 1)[0]
        fracture_center = np.array([self.length / 2, self.length / 2, 0])
        fracture_faces = g_1.cell_centers

        eta = point_pointset(fracture_faces, fracture_center)
        slips = self.analytical_tangential_slip(eta)

        return slips

    def analytical_tangential_slip(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute Westergaard's analytical solution in the shear mode 
        for the tangential displacement jump at given distances from fracture
        cell centers to the fracture center.

        Parameters:
            eta: Distances of fracture cell centers to the fracture center.

        Return
            Array containing analytical tangential displacement jumps.

        """
        # See [2], Ch 2.4 
        cons = 2.0 * (1.0 - self.poi) / self.shear_modulus * self.tau_inf * self.a
        return cons * np.sqrt(1.0 - np.power(eta / self.a, 2))

    def _get_bc_val_from_tangential_displacement(
        self,
        sd: pp.Grid,
        bound_faces: np.ndarray,
        xf: np.ndarray,   
        h: float,         
        dv: float, 
    ) -> np.ndarray:
        """
        Computes semi-analytical displacement values on the boundary using the BEM
        for the Westergaard's problem.

        Parameter
            sd: The matrix grid.
            bound_faces: Array of indices of boundary faces of ``sd``.
            xf: Coordinates of boundary faces.
            h: BEM segment length.
            du: Westergaard's analytical relative tangential displacement.

        Return:
            Boundary values for the displacement.
        """
        u = np.zeros((sd.dim, sd.num_faces))
        alpha = np.pi / 2 - self.theta_rad
        nu = self.poi

        # See [1], Ch 4.5 Coordinate transformations eq. (4.5.4) and eq. (4.5.5)
        m = 1.0 / (4.0 * np.pi * (1.0 - nu))
        f2 = m * (np.log(np.sqrt((xf[0, :] - h) ** 2 + xf[1, :] ** 2))
                  - np.log(np.sqrt((xf[0, :] + h) ** 2 + xf[1, :] ** 2)))
        f3 = -m * (np.arctan2(xf[1, :], (xf[0, :] - h)) - np.arctan2(xf[1, :], (xf[0, :] + h)))
        f4 = m * ( xf[1, :] / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
                 - xf[1, :] / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2))
        f5 = m * ( (xf[0, :] - h) / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
                 - (xf[0, :] + h) / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2))

        # See [1], Ch 4.5 Coordinate transformations eq. (4.5.1) 
        # and Ch5.2 Displacement discontinuity eq. (5.2.3)
        u[0, bound_faces] = dv * (
            -(1 - 2 * nu) * np.sin(alpha) * f2
            + 2 * (1 - nu) * np.cos(alpha) * f3
            + xf[1, :] * (np.sin(alpha) * f4 - np.cos(alpha) * f5)
        )
        u[1, bound_faces] = dv * (
            +(1 - 2 * nu) * np.cos(alpha) * f2
            + 2 * (1 - nu) * np.sin(alpha) * f3
            - xf[1, :] * ( np.cos(alpha) * f4 + np.sin(alpha) * f5 )
        )
        return u

    def displacement_bc_values_from_bem(
        self,
        grid: pp.Grid,
        h: float,
        bound_faces: np.ndarray,
        bem_centers: np.ndarray,
        v_a: np.ndarray,              
    ) -> np.ndarray:
        """
        Computes semi-analytical displacement values using the BEM for the Westergaard's
        problem in the shear mode.

        Parameter
            grid: The matrix grid.
            h: BEM segment length.
            bound_faces: Array of indices of boundary faces of ``grid``.
            bem_centers: BEM segments centers.
            v_a: Westergaard's analytical relative tangential displacement.

        Return:
            Semi-analytical boundary displacement values.

        """
        bc_val = np.zeros((grid.dim, grid.num_faces))
        bound_face_centers = grid.face_centers[:, bound_faces]

        for i in range(0, v_a.size):
            new_bound_face_centers = self.tranform_boundary_face_coordinates_bem(
                bem_centers[:, i], bound_face_centers
            )
            u_bound = self._get_bc_val_from_tangential_displacement(
                grid, bound_faces, new_bound_face_centers, h, v_a[i]
            )
            bc_val += u_bound

        return bc_val

@dataclass
class WestergaardShearModeSaveData:
    """Data class for storing the error in the displacement field."""
    error_tangential_displacement: pp.number


class WestergaardShearModeDataSaving(ManuSneddonDataSaving):
    """Class for saving the error in the displacement field."""

    exact_sol: WestergaardShearModeExactSolution2d

    def collect_data(self) -> WestergaardShearModeSaveData:
        """Collecting the error in the displacement field."""
        frac_sd = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(frac_sd)

        # Computing the numerical displacement jump along the fracture on the fracture.
        # cell centers.
        u_n = (nd_vec_to_tangential @ self.displacement_jump(frac_sd)).value(
            self.equation_system
        )

        # Checking convergence specifically on the fracture.
        u_a = self.exact_sol.exact_sol_tangential_fracture(self.mdg)

        # Truncate the solution at the fracture cells close to the boundary.
        # How close is determined by the parameter eps (percent).
        eps = self.params["error_exclusion_zone_fracture_tips"]
        close_to_fracture_tips = self._index_fracture_cells_close_to_fracture_tips(eps)
        u_a[close_to_fracture_tips] = 0
        u_n[close_to_fracture_tips] = 0

        e = ConvergenceAnalysis.lp_error(
            frac_sd[0], u_a, u_n, is_scalar=False, is_cc=True, relative=True
        )

        collect_data = WestergaardShearModeSaveData(error_tangential_displacement=e)
        return collect_data

class WestergaardShearModeConstitutiveLaws(
    pp.constitutive_laws.CoulombFrictionBound,
    pp.constitutive_laws.LinearElasticMechanicalStress, 
):
    """Linear elastic stress tensor.

    To be used in mechanical problems, e.g. force balance. 
    This does not involve forces from poromechanics (fluid pressure)

    """
    def stress_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpsaAd:
        """Discretization class for the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            The MPSA discretization in Ad operator form.
        """
        return pp.ad.MpsaAd(self.stress_keyword, subdomains)
    
    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction coefficient [-].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Friction coefficient operator.
        """
        mu = float(self.params.get("mu", 0.0))
        n_cells = sum(g.num_cells for g in subdomains)
        vals = mu * np.ones(n_cells)
        return pp.ad.DenseArray(vals, name="friction_coefficient")


class WestergaardShearMode2d(
    ManuSneddonGeometry2d,
    WestergaardShearModeDataSaving,
    ManuSneddonBoundaryConditions,
    WestergaardShearModeConstitutiveLaws,
    pp.MomentumBalance,
):
    """Complete Westergaard shear model by MPSA in PorePy."""

    def __init__(self, params=None):
        super().__init__(params)
        self.exact_sol = WestergaardShearModeExactSolution2d(self)
