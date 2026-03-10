from __future__ import annotations

import numpy as np
import porepy as pp

class GridsPerturbationMixin(pp.PorePyModel):
    """
    A mixin for adding perturbations into the mesh to make simplex grids 
    have some randomness in the 2D matrix. 

    Perturbation is applied based on the following equtions:
    - delta_x = rho * cell_size ^ beta  * ei
    - delta_y = rho * cell_size ^ beta  * ei
    where rho is the perturbation amplitude in [0, 1], beta is the perturbation exponent, 
    and ei is a bounded random variable in [-1, 1].

    This mixin overrides the geometry construction by implementing a custom method 
    for :meth:`set_geometry()` . 
    """
    @staticmethod
    def _get_sparse(grid, name):
        """
        Retrieve the sparse matrix from a PorePy grid. 

        Parameters:
        grid: pp.Grid
            The grid from which to retrieve the sparse matrix.
        name: str
            The name of the sparse matrix attribute to retrieve.

        Returns:
        scipy.sparse matrix
            The requested sparse matrix.
        """    
        obj = getattr(grid, name, None)
        if obj is None:
            raise RuntimeError(f"{name} not found on grid")
        return obj() if callable(obj) else obj

    def set_geometry(self):

        super().set_geometry()

        mdg = self.mdg
        nd = self.nd  
        sd_old = mdg.subdomains(dim=nd)[0]
        sd_new = sd_old.copy()

        # Get boundary nodes
        face_nodes = self._get_sparse(sd_new, "face_nodes")

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