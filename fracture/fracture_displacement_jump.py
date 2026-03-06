"""
Utilities: Fracture tip singularity convergence analysis.

Only valid for one single fracture. 

Sample normal/tangential displacement jumps along a 1D fracture (embedded in a 2D model) and compute
distances from sampling cells to the nearest tip.

"""
from __future__ import annotations

import numpy as np

class FractureDisplacementJump: 
    """
    This class depends on a PorePy model that has already built a mixed-dimensional
    grid and solved for displacements. It includes methods to extract 
    normal/tangential displacement jumps on the fracture and to sample them at cells 
    a specified number of steps away from each tip.

    This implementation assumes a single fracture represented by a 1D subdomain embedded
    in a 2D matrix grid. 

    Attributes:
        model (object): the PorePy model instance providing ``mdg``, operators, and parameters.
        params (dict): the model's parameters dictionary. 
        mdg (pp.MixedDimensionalGrid): mixed-dimensional grid of the model.
        equation_system (pp.EquationSystem): equation system/solution used to evaluate operators.
        sd_frac (pp.Grid): the 1D fracture subdomain grid.
        frac_cell_centers (np.ndarray): ``shape=(dim, frac_num_cells)``
            coordinates of fracture cell centers. 
        frac_num_cells (int): number of fracture cells.  
        tip_faces (np.ndarray): indices of boundary faces that correspond to fracture tips.
        tip_cells (np.ndarray): cell indices adjacent to each tip face.
        tip_face_centers (np.ndarray): ``shape=(dim, n_tips)``
            coordinates of the fracture tip face centers.
    """

    def __init__(
        self, 
        model: object | None = None,
        cache: dict | None = None,
    ) -> None:
        """
        Initialize from a solved model.

        Parameters:
            model: A solved PorePy model instance. 

        Notes: 
            The class is designed as the primary interface for accessing geometry, parameters, 
            and displacement-related solutions from a solved model. 

            All variables from the model are initialized here to provide a consistent interface 
            for other classes inheriting from this class. 
        """
        if model is not None:
            self._initialize_from_model(model)
        elif cache is not None:
            self._initialize_from_cache(cache)
        else:
            raise ValueError("Either a model or a cache must be provided.")
        
    def _initialize_from_model(self, model: object) -> None:
        self.model = model

        self.params = model.params
        self.mdg = model.mdg
        self.equation_system = model.equation_system

        self.sd_frac = self.mdg.subdomains(dim=1)[0]
        self.frac_cell_centers = self.sd_frac.cell_centers
        self.frac_num_cells = self.sd_frac.num_cells
        self.tip_faces = self.sd_frac.tags["tip_faces"].nonzero()[0]                 
        _, self.tip_cells = self.sd_frac.signs_and_cells_of_boundary_faces(self.tip_faces)  
        self.tip_face_centers = self.sd_frac.face_centers[:, self.tip_faces] 

        u_jump = self.model.displacement_jump([self.sd_frac])                     
        normal_op = self.model.normal_component([self.sd_frac]) 
        tangential_op = self.model.tangential_component([self.sd_frac]) 

        self.u_jump_normal = (normal_op @ u_jump).value(self.equation_system)
        self.u_jump_tangential = (tangential_op @ u_jump).value(self.equation_system)
    
    def _initialize_from_cache(self, cache: dict) -> None:
        self.params = cache["params"]

        self.frac_cell_centers = cache["frac_cell_centers"]
        self.frac_num_cells = self.frac_cell_centers.shape[1]
        self.tip_cells = np.array([0, self.frac_num_cells - 1])
        self.tip_face_centers = cache["tip_face_centers"]

        self.u_jump_normal = cache["u_jump_normal"]
        self.u_jump_tangential = cache["u_jump_tangential"]
    
    def normal_displacement_jump(self) -> np.ndarray:
        """
        Returns:
            the normal component of the displacement jump on the fracture.

        """
        return self.u_jump_normal
    
    def tangential_displacement_jump(self) -> np.ndarray:
        """
        Returns:
            the tangential component of the displacement jump on the fracture.

        """
        return self.u_jump_tangential

    def sampling_frac_cell(self, n_steps: int) -> np.ndarray:
        """Compute fracture cell indices to sample 

        For each tip, the selected cell index is computed for both tips.

        Returns:
            current_frac_cells: Integer indices of the fracture cells to sample (shape (n_tips,)).
        """
        cells_tip_one = n_steps
        cells_tip_two = self.tip_cells - n_steps      
        current_frac_cells = np.where(self.tip_cells == 0, cells_tip_one, cells_tip_two)
        
        return current_frac_cells
    
    def sampling_multi_frac_cells(self, n_steps: int) -> np.ndarray:
        """Compute multiple fracture cell indices

        """
        num_frac = self.get_number_of_fracture_cells()
        n_steps = np.asarray(n_steps, dtype=int).ravel()
        cells_tip_one = n_steps
        cells_tip_two = num_frac - n_steps - 1  
        current_frac_cells = np.vstack([cells_tip_one, cells_tip_two])
        
        return current_frac_cells
    
    def extract_sampling_normal_jump(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample normal displacement jumps and distances at selected fracture cells of two tips.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``u_jump``: normal displacement jump values at sampled cells.
                - ``r_cell_to_tip``: distances from sampled cell centers to tip face centers.
        """
        current_frac_cells = self.sampling_frac_cell(n_steps)
        r_cell_to_tip = np.linalg.norm(
            self.tip_face_centers - self.frac_cell_centers[:, current_frac_cells], axis=0
        )
        u_jump = self.normal_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    def extract_sampling_tangential_jump(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample tangential displacement jumps and distances at selected fracture cells of two tips.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``u_jump``: tangential displacement jump values at sampled cells.
                - ``r_cell_to_tip``: distances from sampled cell centers to tip face centers.
        """
        current_frac_cells = self.sampling_frac_cell(n_steps)
        r_cell_to_tip = np.linalg.norm(
            self.tip_face_centers - self.frac_cell_centers[:, current_frac_cells], axis=0
        )
        u_jump = self.tangential_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    def get_number_of_fracture_cells(self) -> int: 
        """
        Return the number of cells on the 1D fracture subdomain.

        This assumes a single fracture (the 1D subdomain in the model).

        Returns:
            int: Number of 1D fracture cells.

        Notes:
            This method should be deleted later.
        """
        # Number of fracture cells
        num_frac_cells = self.frac_num_cells

        return num_frac_cells
    
    def generate_sampling_pairs(self):
        k_min = 0
        k_max = self.frac_num_cells // 2 - 1

        pairs_in_cellsizes = []
        for k in range(k_min, k_max+1, 1):
            pairs_in_cellsizes.append([k, k + 1])

        return pairs_in_cellsizes
    
    def generate_samples_on_half_fracture(self):
        k_min = 0
        k_max = self.frac_num_cells // 2 - 1

        sampling_points = [] 
        for k in range(k_min, k_max+1, 1):
            sampling_points.append([k])

        return sampling_points
    
    def sampling_pairs_by_exponent(self):
        tip = self.tip_face_centers[:, 0]
        r_cell_to_tip = np.linalg.norm(
            tip[:, None] - self.frac_cell_centers, axis=0
        )
        radial_dis = self.radial_dis_by_exponent()
        k_selected = np.where(r_cell_to_tip >= radial_dis)[0]
        sampling_pair = [k_selected[0], k_selected[0] + 1]

        return sampling_pair
    
    def radial_dis_by_exponent(self):

        constant_c = self.params["constant_c"]
        exponent_alpha = self.params["exponent_alpha"]
        h = self.params["meshing_arguments"]["cell_size"]
        a = self.params["a"]
        radial_dis = constant_c * (a ** (1 - exponent_alpha)) * (h  ** exponent_alpha)

        return radial_dis
    
    def extract_multi_points_normal_jump(
        self, 
        sampling_points: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample normal displacement jumps and distances at selected fracture cells of two tips.

        """
        sampling_points = np.array(sampling_points, dtype=int)
        current_frac_cells = self.sampling_multi_frac_cells(sampling_points)

        print("current_frac_cells:", current_frac_cells)
        print("self.tip_face_centers:", self.tip_face_centers)

        tip_one, tip_two = self.tip_face_centers.T
        cells_left = current_frac_cells[0]
        cells_right = current_frac_cells[1]
        coords_left = self.frac_cell_centers[:, cells_left]
        coords_right = self.frac_cell_centers[:, cells_right]

        r_left = np.linalg.norm(tip_one[:, None] - coords_left, axis=0)
        r_right = np.linalg.norm(tip_two[:, None] - coords_right, axis=0)

        r_cell_to_tip = np.vstack([r_left, r_right])
        u_jump = self.normal_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip


        
    