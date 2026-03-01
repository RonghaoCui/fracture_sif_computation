"""
Utilities: Fracture tip singularity convergence analysis.

Only valid for one single fracture. 

Sample normal/tangential displacement jumps along a 1D fracture (embedded in a 2D model) and compute
distances from sampling cells to the nearest tip.

"""
from __future__ import annotations

import porepy as pp
import numpy as np

class FractureDisplacementJump: 
    """
    This class depends on a PorePy model that has already built a mixed-dimensional
    grid (``mdg``) and solved for displacements. It exposes convenience methods to
    extract normal/tangential displacement jumps on the fracture and to sample them
    at cells a specified number of steps away from each tip.

    Attributes:
        model: The PorePy model object providing ``mdg``, operators, and parameters.
        mdg: Mixed-dimensional grid of the model.
        equation_system: Equation system/solution holder used to evaluate operators.
        sd_frac: The 1D fracture subdomain grid (first and only one assumed).
        tip_faces: Indices of boundary faces that correspond to fracture tips.
        tip_cells: Cell indices adjacent to each tip face.
        tip_face_centers: Coordinates of the fracture tip face centers (2 x n_tips).
        params: Reference to the model's parameters dict; expected to include a
            ``"sampling"`` section (see :meth:`sampling_frac_cell`).
    """
    # This class only can be done after having a mdg provided by the Model class (ManuSneddonModel2d here)
    def __init__(self, model: object) -> None:
        """Initialize from a solved model.

        Args:
            model: A solved PorePy model instance

        """
        self.model = model
        self.mdg = model.mdg
        self.equation_system = model.equation_system

        self.sd_frac = self.mdg.subdomains(dim=1)[0]
        self.tip_faces = self.sd_frac.tags["tip_faces"].nonzero()[0]                 
        _, self.tip_cells = self.sd_frac.signs_and_cells_of_boundary_faces(self.tip_faces)  
        self.tip_face_centers = self.sd_frac.face_centers[:, self.tip_faces] 

        self.params = model.params
    
    def normal_displacement_jump(self) -> np.ndarray:
        """
        Return the normal component of the displacement jump on the fracture.

        """
        sd_frac = self.sd_frac
        u_jump = self.model.displacement_jump([sd_frac])                     
        normal_op = self.model.normal_component([sd_frac]) 
        u_jump_whole_frac = (normal_op @ u_jump).value(self.equation_system)
        return u_jump_whole_frac
    
    def tangential_displacement_jump(self) -> np.ndarray:
        """
        Return the tangential component of the displacement jump on the fracture.

        """
        sd_frac = self.sd_frac
        u_jump = self.model.displacement_jump([sd_frac])                     
        tangential_op = self.model.tangential_component([sd_frac]) 
        u_jump_whole_frac = (tangential_op @ u_jump).value(self.equation_system)
        return u_jump_whole_frac

    def sampling_frac_cell(self, n_steps: int) -> np.ndarray:
        """Compute fracture cell indices to sample based on ``params['sampling']``.

        Expects ``self.params`` to contain a ``"sampling"`` configuration with one of:
            - ``{"mode": "refinement", "level": L}`` → uses ``n_steps = 2 * L``;
            - ``{"mode": "fixed", "steps": k}`` → uses exactly ``n_steps = k``.

        For each tip, the selected cell index is computed for both tips.

        Returns:
            np.ndarray: Integer indices of the fracture cells to sample (shape (n_tips,)).

        Raises:
            ValueError: If sampling mode is missing/unknown or required keys are absent.
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
        current_frac_cells = np.vstack([cells_tip_one, cells_tip_two], dtype=int)
        
        return current_frac_cells
    
    def extract_sampling_normal_jump(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample normal displacement jumps and distances at selected fracture cells of two tips.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``u_jump``: normal displacement jump values at sampled cells.
                - ``r_cell_to_tip``: distances from sampled cell centers to tip face centers.
        """
        sd_frac = self.sd_frac
        current_frac_cells = self.sampling_frac_cell(n_steps)
        r_cell_to_tip = np.linalg.norm(
            self.tip_face_centers - sd_frac.cell_centers[:, current_frac_cells], axis=0
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
        sd_frac = self.sd_frac
        current_frac_cells = self.sampling_frac_cell(n_steps)
        r_cell_to_tip = np.linalg.norm(
            self.tip_face_centers - sd_frac.cell_centers[:, current_frac_cells], axis=0
        )
        u_jump = self.tangential_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    def get_number_of_fracture_cells(self) -> int: 
        """
        Return the number of cells on the 1D fracture subdomain.

        This assumes a single fracture (the 1D subdomain in the model).

        Returns:
            int: Number of 1D fracture cells.
        """
        # Number of fracture cells
        num_frac_cells = self.sd_frac.num_cells

        return num_frac_cells
    
    def generate_sampling_pairs(self):
        num_frac_cells = self.sd_frac.num_cells
        k_min = 0
        k_max = num_frac_cells // 2 - 1

        pairs_in_cellsizes = []
        
        for k in range(k_min, k_max+1, 1):
            pairs_in_cellsizes.append([k, k + 1])

        return pairs_in_cellsizes
    
    # def generate_sampling_three(self):
    #     num_frac_cells = self.sd_frac.num_cells
    #     k_min = 1
    #     k_max = num_frac_cells // 2 - 1
    #     three_in_cellsizes = []
    #     for k in range(k_min-1, k_max+1, 1):
    #         three_in_cellsizes.append([k - 1, k, k + 1])

    #     return three_in_cellsizes
    
    def generate_samples_on_half_fracture(self):
        num_frac_cells = self.sd_frac.num_cells
        k_min = 0
        k_max = num_frac_cells // 2 - 1

        sampling_points = []
        
        for k in range(k_min, k_max+1, 1):
            sampling_points.append([k])

        return sampling_points
    
    def sampling_pairs_by_exponent(self):

        sd_frac = self.sd_frac
        tip = self.tip_face_centers[:, 0]
        r_cell_to_tip = np.linalg.norm(
            tip[:, None] - sd_frac.cell_centers, axis=0
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
    
    def extract_multi_points_normal_jump(self, sampling_points: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample normal displacement jumps and distances at selected fracture cells of two tips.

        """
        sampling_points = np.array(sampling_points, dtype=int)
        sd_frac = self.sd_frac
        current_frac_cells = self.sampling_multi_frac_cells(sampling_points)
        print("current_frac_cells:", current_frac_cells)
        print("self.tip_face_centers:", self.tip_face_centers)
        tip_one, tip_two = self.tip_face_centers.T
        cells_left = current_frac_cells[0]
        cells_right = current_frac_cells[1]
        coords_left = sd_frac.cell_centers[:, cells_left]
        coords_right = sd_frac.cell_centers[:, cells_right]

        r_left = np.linalg.norm(tip_one[:, None] - coords_left, axis=0)
        r_right = np.linalg.norm(tip_two[:, None] - coords_right, axis=0)

        r_cell_to_tip = np.vstack([r_left, r_right])
        u_jump = self.normal_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip


        
    