from __future__ import annotations

import numpy as np
import porepy as pp

class GridsPerturbationMixin(pp.PorePyModel):
    """
    A mixin for introducing random perturbations to simplex grids (triangles) 
    in the 2D matrix. 

    Perturbation is applied based on the following equtions:
        - delta_x = rho * cell_size ^ beta  * ei
        - delta_y = rho * cell_size ^ beta  * ei
        where rho is the perturbation amplitude in [0, 1], beta is the perturbation exponent, 
        and ei is a bounded random variable in [-1, 1].

    Notes: 
        This mixin overrides the geometry construction by implementing a custom method 
        for :meth:`set_geometry()` . 

    """
    def set_geometry(self) -> None:
        """
        Construct the model geometry and then perturb the grid nodes randomly. 
        
        The perturbation is applied to the interior nodes of grids in the 2D matrix, 
        while keeping the boundary and fracture interface nodes fixed.
        
        The perturbation parameters are defined in the model parameters provided in 
        ``self.params``, and include: 
            ``perturb_amplitude``: A value in [0, 1] that controls the maximum relative 
                    perturbation magnitude.
            ``perturb_exponent``: A positive value that controls how the perturbation scales 
                    with the local cell size. 
            ``perturb_seed``: An integer seed for the random number generator to ensure 
                    reproducibility of the perturbation.

        Raises: 
            ValueError: If the perturbation results in any inverted cells 
                (i.e., cells with non-positive volume).
        """
        super().set_geometry()

        mdg = self.mdg
        nd = self.nd  
        sd_old = mdg.subdomains(dim=nd)[0]
        sd_new = sd_old.copy()

        # Get boundary nodes
        face_nodes = sd_new.face_nodes

        boundary_faces = np.where(sd_new.tags["domain_boundary_faces"])[0]
        boundary_nodes = np.unique(face_nodes[:, boundary_faces].indices)

        # Get fracture interface nodes
        interface_faces = np.where(sd_new.tags["fracture_faces"])[0]
        interface_nodes = np.unique(face_nodes[:, interface_faces].indices)

        fixed_nodes = np.union1d(boundary_nodes, interface_nodes)

        # Nodes that can be perturbed (not on boundaries or interfaces)
        all_nodes = np.arange(sd_new.num_nodes)
        perturb_nodes = np.setdiff1d(all_nodes, fixed_nodes, assume_unique=False)

        # Perturbation parameters
        cell_size = self.params["meshing_arguments"]["cell_size"]
        amp_rel = self.params["perturb_amplitude"]
        perturb_exponent = self.params["perturb_exponent"]

        amp_abs = amp_rel * cell_size ** perturb_exponent

        rng = np.random.default_rng(self.params["perturb_seed"])

        # Fracture orientation
        theta = self.params["theta_rad"]
        t = np.array([np.sin(theta), np.cos(theta)], dtype=float)   # tangent to the fracture
        n = np.array([np.cos(theta), -np.sin(theta)], dtype=float)  # normal to the fracture

        # Perturbation in both tangential and normal directions
        xi_t = rng.uniform(-1, 1, size=perturb_nodes.size)
        yi_t = rng.uniform(-1, 1, size=perturb_nodes.size)

        jt = amp_abs * xi_t
        jn = amp_abs * yi_t

        dx = jt * t[0] + jn * n[0]
        dy = jt * t[1] + jn * n[1]

        sd_new.nodes[0, perturb_nodes] += dx
        sd_new.nodes[1, perturb_nodes] += dy

        # Recompute geometry
        sd_new.compute_geometry()
        if np.any(sd_new.cell_volumes <= 0):
            raise ValueError(
                "Cell inversion detected after perturbation. "
            )

        # Update the mdg with the new subdomain
        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

    def check_acute_triangles(self, sd: pp.Grid) -> tuple[bool, int]:
        """
        Check whether all triangles in the given grid are acute, and also return
        the number of non-acute triangles if any. 

        Parameters:
            sd: the subdomain to check. 

        Returns:
            tuple[bool, int]: 
                bool: True if all triangles are acute, False otherwise.
                int: The number of non-acute triangles in the grid.

        Raises: 
            ValueError: If any cell in the grid is not a triangle 
                i.e., does not have exactly 3 nodes.

        """
        nodes = sd.nodes[:2]
        cell_to_node_mat = sd.cell_nodes()
        
        num_nonacute = 0

        for i in range(sd.num_cells):
            node_indices_in_cell = cell_to_node_mat[:, i].indices

            if len(node_indices_in_cell) != 3:
                raise ValueError(f"Cell {i} is not a triangle.")

            A = nodes[:, node_indices_in_cell[0]]
            B = nodes[:, node_indices_in_cell[1]]
            C = nodes[:, node_indices_in_cell[2]]

            AB = B - A
            AC = C - A
            BA = A - B
            BC = C - B
            CA = A - C
            CB = B - C 

            condA = np.dot(AB, AC) > 0
            condB = np.dot(BA, BC) > 0
            condC = np.dot(CA, CB) > 0 

            if not (condA and condB and condC):
                num_nonacute += 1
        
        return num_nonacute == 0, num_nonacute

