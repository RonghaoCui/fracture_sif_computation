"""
Model for Sneddon's problem by a quadratic finite element method with quarter-point elements
using Sneddon's analytical solution for normal displacement jump.

FEM code resource:
    - [1] Dang et al., International Journal of Rock Mechanics and Mining Sciences, 160 (2022): 105248.

"""
from __future__ import annotations

import porepy as pp
import numpy as np

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d
from fems import mmfa

import matplotlib.pyplot as plt

import os
save_dir = "figures/benchmarks/femqpe/sol"
os.makedirs(save_dir, exist_ok=True)

# Directly use boundary element methods in Sneddon problem
class ExactDirichletBCSneddon:
    """Class representing the analytical solution for the pressurized fracture
    problem."""

    def __init__(self, params: dict) -> None:
        self.params = params
        solid = self.params["material_constants"]["solid"]
        self.mu = solid.shear_modulus          # μ
        self.lam = solid.lame_lambda           # λ
        self.poi = self.lam / 2.0 / (self.mu + self.lam)    # possion ratio
        self.E = 2.0 * self.mu * (1.0 + self.poi)           # Young's modulus
        self.p0 = self.params["p0"]
        self.theta_rad = params["theta_rad"]
        self.a = self.params["a"]
        self.length = params["domain_size"]
        self.n = params["num_bem_segments"]

    def analytical_displacements(self, eta: np.ndarray) -> np.ndarray:
        """
        Sneddon analytical normal displacement jump.
        """
        cons = (1.0 - self.poi) / self.mu * self.p0 * self.a * 2.0
        return cons * np.sqrt(1.0 - (eta / self.a) ** 2)

    def get_bem_centers(self, h: float, center: np.ndarray) -> np.ndarray:
        """
        Coordinates of BEM segment centers.
        """
        bem_centers = np.zeros((3, self.n))
        x0 = center[0] - (self.a - 0.5 * h) * np.sin(self.theta_rad)
        y0 = center[1] - (self.a - 0.5 * h) * np.cos(self.theta_rad)

        i = np.arange(self.n)
        bem_centers[0] = x0 + i * h * np.sin(self.theta_rad)
        bem_centers[1] = y0 + i * h * np.cos(self.theta_rad)

        return bem_centers

    def transform(self, xc: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Transform coordinates to local BEM system.
        """
        alpha = np.pi / 2.0 - self.theta_rad
        x_bar = np.zeros_like(x)

        x_bar[0] = (x[0] - xc[0]) * np.cos(alpha) + (x[1] - xc[1]) * np.sin(alpha)
        x_bar[1] = -(x[0] - xc[0]) * np.sin(alpha) + (x[1] - xc[1]) * np.cos(alpha)

        return x_bar

    def bem_kernel_displacement(
        self, xloc: np.ndarray, h: float, du: float
    ) -> tuple[float, float]:
        """
        Displacement contribution (ux, uy) at a point from one BEM segment.
        """
        xf, yf = xloc
        
        alpha = np.pi / 2.0 - self.theta_rad

        m = 1.0 / (4.0 * np.pi * (1.0 - self.poi))

        f2 = m * (
            np.log(np.sqrt((xf - h) ** 2 + yf**2))
            - np.log(np.sqrt((xf + h) ** 2 + yf**2))
        )

        f3 = -m * (
            np.arctan2(yf, xf - h) - np.arctan2(yf, xf + h)
        )

        f4 = m * (
            yf / ((xf - h) ** 2 + yf**2)
            - yf / ((xf + h) ** 2 + yf**2)
        )

        f5 = m * (
            (xf - h) / ((xf - h) ** 2 + yf**2)
            - (xf + h) / ((xf + h) ** 2 + yf**2)
        )

        ux = du * (
            -(1.0 - 2.0 * self.poi) * np.cos(alpha) * f2
            - 2.0 * (1.0 - self.poi) * np.sin(alpha) * f3
            - yf * (np.cos(alpha) * f4 + np.sin(alpha) * f5)
        )

        uy = du * (
            -(1.0 - 2.0 * self.poi) * np.sin(alpha) * f2
            + 2.0 * (1.0 - self.poi) * np.cos(alpha) * f3
            - yf * (np.sin(alpha) * f4 - np.cos(alpha) * f5)
        )

        return ux, uy

    def displacement_at_point(self, x: float, y: float) -> tuple[float, float]:
        """
        Return analytical displacement (ux, uy) at a point (x, y).
        """
        h = 2.0 * self.a / self.n
        center = np.array([self.length / 2.0, self.length / 2.0, 0.0])

        bem_centers = self.get_bem_centers(h, center)

        eta = np.linalg.norm(
            bem_centers[:2] - center[:2, None], axis=0
        )
        u_open = self.analytical_displacements(eta)

        ux, uy = 0.0, 0.0
        xg = np.array([x, y, 0.0])

        for i in range(self.n):
            xloc = self.transform(bem_centers[:, i], xg)
            du = -u_open[i]
            dux, duy = self.bem_kernel_displacement(xloc[:2], h / 2.0, du)
            ux += dux
            uy += duy

        return ux, uy

# Class for the FEM model with qpe in Sneddon's problem
class SneddonFemQpe2dSolver:
    """
    Sneddon model by the finite element method with quarter-point elements.
    
    """
    def __init__(self, params: dict, use_qpe: bool) -> None:
        """
        Initialize elastic constants from the input parameters.

        Parameters:
            params: dictionary of model parameters.

        """
        self.params = params
        self.use_qpe = use_qpe
        solid = self.params["material_constants"]["solid"]
        self.mu = solid.shear_modulus          # μ
        self.lam = solid.lame_lambda           # λ
        self.poi = self.lam / 2.0 / (self.mu + self.lam)    # possion ratio
        self.E = 2.0 * self.mu * (1.0 + self.poi)           # Young's modulus
        self.p0 = self.params["p0"]
        self.theta_rad = params["theta_rad"]
        self.a = self.params["a"]
        self.length = params["domain_size"]
        self.cell_size = self.params["meshing_arguments"]["cell_size"]
        self.tol = 1e-6
        self.frac_gap = self.tol * self.cell_size
        model = ManuSneddonModel2d(params=self.params)
        model.prepare_simulation()
        self.grid2d = model.mdg.subdomains(dim=2)[0]
        sd_frac = model.mdg.subdomains(dim=1)[0]
        self.num_frac_cells = sd_frac.num_cells
        tip_faces = sd_frac.tags["tip_faces"].nonzero()[0]
        self.tip_face_centers = sd_frac.face_centers[:2, tip_faces].T   # only [x, y] and shape = (n_tips, 2)
        self.nodes_mdg = None
        self.elements_mdg = None
        self.nodes_p6 = None
        self.elements_t6 = None
        self.elements_quarter = None
        self.U = None

    def model_setup_and_sol(self) -> np.ndarray:
        """
        This is the main part of the FEM solver. It follows the steps as below: 
            1. Get fractured mesh from the matrix mesh for fractured domain from the PorePy existing model. 
            2. Get quardratic triangles with 6 nodes with/without quarter-point elements (QPE) for fracture tips. 
            3. Assemble Dirichlet and Neumann boundary conditions.
            4. Get traction forces for nodes based on Neumann traction on edges.
            5. Get the stiffness matrix.
            6. Solve the displacement solution from the linear system: Ku=F.
        
        Returns: 
            U: shape = (n, 1)
                the solution of displacement components ux and uy.

        """
        # nodes_mdg: node coordinates: np.array(shape = (n, 2), dtype = float)
        # elements_mdg: nodes from each triangle: np.array(shape = (n, 3), dtype = int)
        # Note: frac_gap is the assumed distance between two fracture faces. It should be very small. 
        self.nodes_mdg, self.elements_mdg = mmfa.adjustmesh(self.grid2d, self.tip_face_centers, self.frac_gap)
        # print("tip_face_centers: ", tip_face_centers)

        # nodes_p6: node coordinates, np.array(shape = (n, 2), dtype = float)
        # elements_t6: node connection in cells, np.array(shape = (n, 6), dtype = int)
        # elements_quarter: node connection in cells with fractrue tip nodes, list: shape = (n, 6)
        self.nodes_p6, self.elements_t6, self.elements_quarter = mmfa.t3tot6(
            self.nodes_mdg, 
            self.elements_mdg, 
            tips=self.tip_face_centers, 
            QPE=self.use_qpe
        )
        # x = self.nodes_p6[:, 0]
        # y = self.nodes_p6[:, 1]

        # print("min |y - 0.5|:", np.min(np.abs(y - 0.5)))
        # print("number of nodes near fracture:",
        #     np.sum(np.abs(y - 0.5) < self.frac_gap))
        
        dirdof, dirval = self.Dirichlet_bc_domain()

        NeuDof, NeuVal = self.Neumann_bc_fracture()

        # Assemble matrices for the linear system: Ku=F
        # Assemble the right hand side
        # Get traction forces for nodes based on Neumann traction on edges
        F = mmfa.loadsegment(self.nodes_p6, NeuDof=NeuDof, NeuVal=NeuVal)

        # Plane strain: constitutive law in 2D
        # σ​=2·μe​+λ·​trace(e)·I: e=[exx, eyy, exy].T
        D = np.array([
            [self.lam + 2.0*self.mu, self.lam,               0.0],
            [self.lam,               self.lam + 2.0*self.mu, 0.0],
            [0.0,                    0.0,                    self.mu],
        ], dtype=float)

        if not self.use_qpe:
            self.elements_quarter = []   # No quarter-point elements (QPE) if not using QPE.

        # Assemble the left hand side
        K = mmfa.stiffness(self.nodes_p6, self.elements_t6, D, self.elements_quarter)  

        self.U = mmfa.linearsolver(K, F, dirdof.astype(int), dirval.astype(float), sdof=2*self.nodes_p6.shape[0])  # (1, 2N)

        return self.U
    
    def extract_displacement_jump(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the displacement jump between the conforming fracture surfaces.

        Returns: 
            x_coords: shape = (n, )
                x coordinates of fracture surface nodes from small to large values.            
            u_jump_normal: shape = (n, )
                normal displacement jump along the fracture.
            u_jump_tangential: shape = (n, )
                tangential displacement jump along the fracture.

        """
        # Only for the horizontal fracture
        upper_frac_nodes, lower_frac_nodes = self.fracture_surface_nodes()

        upper_frac_nodes = upper_frac_nodes[np.argsort(self.nodes_p6[upper_frac_nodes, 0])]
        lower_frac_nodes = lower_frac_nodes[np.argsort(self.nodes_p6[lower_frac_nodes, 0])]

        uy = self.U[1::2, 0]  
        ux = self.U[0::2, 0]
        u_jump_normal = uy[upper_frac_nodes] - uy[lower_frac_nodes]
        u_jump_tangential = ux[upper_frac_nodes] - ux[lower_frac_nodes]
        x_coords = 0.5*(self.nodes_p6[upper_frac_nodes, 0] + self.nodes_p6[lower_frac_nodes, 0])

        return x_coords, u_jump_normal, u_jump_tangential

    def Dirichlet_bc_domain(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Set up types and values of Dirichlet boundary conditions for domain boundaries.

        Returns: 
            dirdof: indices of DOFs (ux, uy) in Dirichlet boundary conditions.             
            dirval: values of Dirichlet boundary conditions corresponding to DOFs.

        """
        # Add Dirichlet boundary conditions for domain
        # Get domain boundaries
        xmin, xmax = np.min(self.nodes_p6[:, 0]), np.max(self.nodes_p6[:, 0])
        ymin, ymax = np.min(self.nodes_p6[:, 1]), np.max(self.nodes_p6[:, 1])

        # Get domain boundary nodes
        left_boundary = np.where(np.abs(self.nodes_p6[:, 0] - xmin) < self.tol)[0]
        right_boundary  = np.where(np.abs(self.nodes_p6[:, 0] - xmax) < self.tol)[0]
        bottom_boundary = np.where(np.abs(self.nodes_p6[:, 1] - ymin) < self.tol)[0]
        top_boundary = np.where(np.abs(self.nodes_p6[:, 1] - ymax) < self.tol)[0]
        # All domain boundary nodes
        domain_boundaries = np.unique(np.concatenate([
            left_boundary, 
            right_boundary, 
            bottom_boundary, 
            top_boundary
        ]))
        
        # Dirichlet DOFs (ux, uy)
        dirdof = np.concatenate([2*domain_boundaries, 2*domain_boundaries + 1])
        dirval = np.zeros_like(dirdof, dtype=float)      # All zero
        # Values of Dirichlet boundary nodes directly get from BEM: ExactDirichletBCSneddon
        for i, dof in enumerate(dirdof):
            node = dof // 2
            component = dof % 2
            x, y = self.nodes_p6[node]
            ux, uy = self.displacement_at_point(x, y)
            dirval[i] = ux if component == 0 else uy
        
        return dirdof, dirval
    
    def Neumann_bc_fracture(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Set up types and values of Neumann boundary conditions for internal boundaries (fracture surfaces).

        Returns: 
            NeuDof: three node indices of edges correponding to traction.             
            NeuVal: values of traction corresponding to edges. 
        
        Notes:
            Tip 1 does not mean the left tip! It is determined by the order of tip_face_centers from the mesh. 
        """
        all_boundaries_t3 = self.boundary_edges_from_t3()
        midnodes_edges = self.get_midnodes_from_edges()
        # Get the edge nodes of fracture surfaces by a preset tolerence
        # Horizontal fracture by default
        # Tip 1 does not mean the left tip! 
        x_tip = self.tip_face_centers[:, 0]
        x_tip_min = np.min(x_tip)
        x_tip_max = np.max(x_tip)

        fracture_edges = []
        for (n0, n1) in all_boundaries_t3:
            x0, y0 = self.nodes_p6[n0]
            x1, y1 = self.nodes_p6[n1]
            xm, ym = 0.5*(x0+x1), 0.5*(y0+y1)

            if (x_tip_min - self.tol) <= xm <= (x_tip_max + self.tol) and abs(ym - self.length/2.0) < 5.0*self.frac_gap:
                fracture_edges.append((n0, n1))

        fracture_edges = np.array(fracture_edges, dtype=int)

        # Get Neumann boundary conditions
        # Formats of Neumann boundary conditions in mmfa.loadsegment: 
        #     NeuDof: [n0, nmid, n1] in one edge with three nodes in T6
        #     NeuVal: [tx, ty] for one edge
        NeuDof = []
        NeuVal = []
        # Here we find three nodes of each boundary edge on fracture surfaces 
        # and give traction values for the edge, instead of nodes. 
        for (n0, n1) in fracture_edges:
            key = tuple(sorted((int(n0), int(n1))))
            if key not in midnodes_edges:
                continue
            m01 = midnodes_edges[key]

            ym = 0.5*(self.nodes_p6[n0,1] + self.nodes_p6[n1,1])
            if ym > self.length/2.0:
                # upper fracture surface
                tr = np.array([0.0, +self.p0], dtype=float)
            else:
                # lower fracture surface
                tr = np.array([0.0, -self.p0], dtype=float)

            NeuDof.append([n0, m01, n1])
            NeuVal.append(tr)

        NeuDof = np.array(NeuDof, dtype=int)
        NeuVal = np.array(NeuVal, dtype=float)

        return NeuDof, NeuVal
    
    # Prepare the set-up of the pressure (Neumann boundary conditions) 
    def boundary_edges_from_t3(self) -> np.ndarray:
        """
        Find the boundary edges in the T3 mesh. 

        Returns: 
            Array of shape (num_boundary_edges, 2) containing the node indices
            of all boundary edges. Edges are identified by the occurence count of an egde in
            a mesh. Each row represents one edge, given by the indices of its two end nodes.

        """
        # elements_mdg: Array of shape=(num_cells, 3) containing the node indices of T3. 
        # Each row corresponds to one triangle.
        e01 = self.elements_mdg[:, [0, 1]]
        e12 = self.elements_mdg[:, [1, 2]]
        e20 = self.elements_mdg[:, [2, 0]]
        edges = np.vstack([e01, e12, e20])
        edges = np.sort(edges, axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        return uniq[counts == 1]

    def get_midnodes_from_edges(self) -> dict[tuple[int, int], int]:
        """
        A mapping from mesh edges to their middle nodes. 

        Returns: 
            A mapping pf each edge to the middle nodes in T6. 

        """
        # Array of shape=(num_cells, 6) containing the node indices of T6. 
        # Each row corresponds to one triangle.
        mp = {}
        for tri in self.elements_t6:
            v0, m01, v1, m12, v2, m02 = tri.tolist()
            mp[tuple(sorted((v0, v1)))] = m01
            mp[tuple(sorted((v1, v2)))] = m12
            mp[tuple(sorted((v0, v2)))] = m02
        return mp
    
    def fracture_surface_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get node indices of upper and lower fracture surfaces (one horizontal central fracture by default). 

        Returns: 
            upper_frac_nodes: node indices on the upper fracture surface
            lower_frac_nodes: node indices on the lower fracture surface
        
        Notes:
            Tip 1 does not mean the left tip! It is determined by the order of tip_face_centers from the mesh.    
        """
        x_tip = self.tip_face_centers[:, 0]
        y_tip = self.tip_face_centers[:, 1]
        x_tip_min = np.min(x_tip)
        x_tip_max = np.max(x_tip)   
        y_frac = (y_tip[0] + y_tip[1]) / 2.0   # horizontal fracture by default
        x = self.nodes_p6[:, 0]
        y = self.nodes_p6[:, 1]

        # Using the preset tolerance and fracture gap with super small values can work here
        nodes_frac = (x >= x_tip_min - self.tol) & (x <= x_tip_max + self.tol) & (np.abs(y - y_frac) < 5.0*self.frac_gap)
    
        upper_frac_nodes = np.where(nodes_frac & (y > y_frac))[0]
        lower_frac_nodes = np.where(nodes_frac & (y < y_frac))[0]

        return upper_frac_nodes, lower_frac_nodes

class FemQpeSneddonModel2d(
    ExactDirichletBCSneddon, 
    SneddonFemQpe2dSolver,
): 
    """
    Mixer class for the FEM model with or without QPE in Sneddon's problem. 
    """
    def __init__(self, params: dict, use_qpe: bool) -> None:
        ExactDirichletBCSneddon.__init__(self, params)
        SneddonFemQpe2dSolver.__init__(self, params, use_qpe)

        