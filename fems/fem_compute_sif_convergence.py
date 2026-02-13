from __future__ import annotations

import porepy as pp
import numpy as np

class FEMComputeSifErrorConvergence:
    """
    FEM analogue of MPSA's SIF error convergence analysis in PorePy.
    """

    def __init__(self, model):
        self.model = model
        self.tip_face_centers = model.tip_face_centers
        self.E = self.model.E
        self.poi = self.model.poi

        if model.U is None:
            raise RuntimeError("Model must be solved before extracting jumps.")

    def FEM_extract_sampling_normal_jump(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the normal component of the displacement jump and corresponding radial distance to the
        closer tip with given sampling locations for two symmetrical tips. 

        Parameters: 
            n_steps: the n-th cell near the tip. 

        Returns: 
            u_jump: shape = (n_tips, )
                the normal component of the displacement jump on the fracture.
            r_cell_to_tip: shape = (n_tips, )
                the radial distance to the closer fracture tip. 

        """
        x_coords, u_jump_normal, _ = self.model.extract_displacement_jump()    
        num_frac = len(x_coords)
        current_frac_cells = self.FEM_sampling_frac_cell(n_steps, num_frac)
        # only [x, y] and shape = (n_tips, 2)
        r_cell_to_tip = np.abs(self.tip_face_centers[:,0] - x_coords[current_frac_cells])
        u_jump = u_jump_normal[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    def FEM_sampling_frac_cell(self, n_steps: int, num_frac: int) -> np.ndarray:
        """
        Get sampling indices of fracture nodes for both tips of one fracture. 

        Parameters: 
            n_steps: the n-th cell near the tip.
            num_frac: the number of fracture cells. 

        Returns: 
            current_frac_cells: shape = (2, )
                sampling indices of fracture nodes for tips 1 and 2 of one fracture.

        Notes: 
            It should be noted that there is no fracture tip node on fracture coords.

        """
        cells_tip_one = n_steps
        cells_tip_two = num_frac - n_steps - 1  
        current_frac_cells = np.array([cells_tip_one, cells_tip_two], dtype=int)
        
        return current_frac_cells
    
    def FEM_two_points_single_scale_normal(
        self, sampling_pair: list
    ) -> tuple[float, np.ndarray]:
        """
        Calculate SIF KI by using the two-point method at one single scale of radial distance.

        Parameters: 
            sampling_pair: two nodes on fracture surfaces selected for two-point methods. 

        Returns: 
            Relative errors of calculated SIFs by FEM for tips, compared with analytical SIF values. 

        """
        u_jump_n_one, r_n_one = self.FEM_extract_sampling_normal_jump(sampling_pair[1])
        u_jump_n_two, r_n_two = self.FEM_extract_sampling_normal_jump(sampling_pair[0])
        
        return self.FEM_TwoPoints_SIF_KI_errors(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
    
    def FEM_TwoPoints_SIF_KI_errors(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray,
        u_jump_n_two: np.ndarray, 
        r_n_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Calculate relative errors of calculated SIFs by FEM for tips, compared with analytical SIF values. 

        Parameters:
            u_jump_n_one: displacement jump of node 1
            r_n_one: radial distance to the tip of node 1
            u_jump_n_two: displacement jump of node 2
            r_n_two: radial distance to the tip of node 2

        Notes: 
            node 1 here is the one which is farther from the fracture tips. 
        """
        singularity_orders = 0.5
        KIs = self.FEM_KI_from_two_points(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two, singularity_orders)
        KI_exact = self.FEM_calculate_exact_SIF_KI_sneddon()

        return self.FEM_relative_L2_error(KIs, KI_exact)
    
    def FEM_KI_from_two_points(
            self, 
            u_jump_n_one: np.ndarray, 
            r_n_one: np.ndarray, 
            u_jump_n_two: np.ndarray, 
            r_n_two: np.ndarray,
            singularity_orders: np.ndarray
    ) -> np.ndarray: 
        """
        Calculate SIFs by FEM for tips based on extracted displacement jump. 
        
        Parameters:
            u_jump_n_one: displacement jump of node 1.
            r_n_one: radial distance to the tip of node 1.
            u_jump_n_two: displacement jump of node 2.
            r_n_two: radial distance to the tip of node 2.
            singularity_orders: singularity exponent, 0.5 by default.

        Returns: 
            KIs: SIF values for tips from FEM.  
        """
        KIs_n_one = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_n_one, singularity_orders)
            * u_jump_n_one
        )

        KIs_n_two = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_n_two, singularity_orders)
            * u_jump_n_two
        )

        KIs = KIs_n_one + r_n_one / (r_n_two - r_n_one) * (KIs_n_one - KIs_n_two)

        return KIs
    
    def FEM_calculate_exact_SIF_KI_sneddon(self) -> float: 
        """
        Calculate the analytical solution of SIF KI for Sneddon's problem.

        Returns: 
            KI_exact: an analytical KI value.

        """
        pres0 = self.model.params["p0"]
        frac_half_length = self.model.params["a"]
        KI_exact =  pres0 * np.sqrt(np.pi * frac_half_length)

        return KI_exact
    
    def FEM_relative_L2_error(
        self, KIs: np.ndarray, KI_exact: float
    ) -> tuple[float, np.ndarray]:
        """
        Compute relative errors of SIF KI against an exact reference.

        Parameters:
            KIs: Estimated KI values per tip.
            KI_exact: Exact KI used as the scalar reference.

        Returns:
            rel_l2_error_ave: relative L2 error averaged by two tips.
            rel_error_per_tip: relative absolute error array from two tips.
            
        """
        rel_l2_error_ave = (
            np.linalg.norm(KIs - KI_exact, ord=2) /
            (KI_exact * np.sqrt(len(KIs)))
        )

        rel_error_per_tip = np.abs(KIs - KI_exact) / KI_exact

        return rel_l2_error_ave, rel_error_per_tip
    
    def FEM_extract_multi_points_normal(
        self, sampling_points: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the normal component of the displacement jump and corresponding radial distance to the
        closer tip with given sampling locations for two symmetrical tips. 

        Parameters: 
            sampling_points: the sampled group of cells near the tip. 

        Returns: 
            u_jump: shape = (n_tips, n_steps)
                the normal component of the displacement jump on the fracture.
            r_cell_to_tip: shape = (n_tips, n_steps)
                the radial distance to the closer fracture tip. 

        Notes:         
            self.tip_face_centers only has [x, y] and shape = (n_tips,)    

        """
        sampling_points = np.array(sampling_points, dtype=int)
        n_steps = sampling_points
        x_coords, u_jump_normal, _ = self.model.extract_displacement_jump()    
        num_frac = len(x_coords)
        current_frac_cells = self.FEM_sampling_multi_frac_cells(n_steps, num_frac)
        r_cell_to_tip = np.abs(self.tip_face_centers[:,0].reshape(2, 1) - x_coords[current_frac_cells])
        u_jump = u_jump_normal[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    def FEM_sampling_multi_frac_cells(
        self, n_steps: np.ndarray, num_frac: int
    ) -> np.ndarray:
        """
        Get sampling indices of fracture nodes for both tips of one fracture. 

        Parameters: 
            n_steps: the sampled group of cells near the tip.
            num_frac: the number of fracture cells. 

        Returns: 
            current_frac_cells: shape = (2, n_steps)
                sampling indices of fracture nodes for tips 1 and 2 of one fracture.

        Notes: 
            It should be noted that there is no fracture tip node on fracture coords.

        """
        cells_tip_one = n_steps
        cells_tip_two = num_frac - n_steps - 1  
        current_frac_cells = np.vstack([cells_tip_one, cells_tip_two], dtype=int)
        
        return current_frac_cells
    
    def FEM_multi_points_single_scale_normal(
        self, sampling_points: list
    ) -> tuple[float, np.ndarray]:
        """
        Calculate SIF KI by using the multi-point method at one single scale of radial distance.

        Parameters: 
            sampling_points: a group of nodes on fracture surfaces selected for multi-point methods. 

        Returns: 
            Relative errors of calculated SIFs by FEM for tips, compared with analytical SIF values. 

        """
        u_jump_n, r_n = self.FEM_extract_multi_points_normal(sampling_points)
        
        return self.FEM_MultiPoints_SIF_KI_errors(u_jump_n, r_n)
    
    def FEM_MultiPoints_SIF_KI_errors(
        self, 
        u_jump_n: np.ndarray, 
        r_n: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Calculate relative errors of calculated SIFs by FEM for tips, compared with analytical SIF values. 

        Parameters:
            u_jump_n: displacement jump of sampled nodes. 
            r_n: radial distance to the tip of sampled nodes.

        """
        singularity_orders = 0.5
        KIs, _ = self.FEM_KI_from_multi_points(u_jump_n, r_n, singularity_orders)
        KI_exact = self.FEM_calculate_exact_SIF_KI_sneddon()

        return self.FEM_relative_L2_error(KIs, KI_exact)
    
    def FEM_KI_from_multi_points(
            self, 
            u_jump_n: np.ndarray, 
            r_n: np.ndarray, 
            singularity_orders: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: 
        """
        Calculate SIFs by FEM for tips based on extracted displacement jump. 
        
        Parameters:
            u_jump_n: displacement jump of sampled nodes.
            r_n: radial distance to the tip of sampled nodes.
            singularity_orders: singularity exponent, 0.5 by default.

        Returns: 
            KIs: SIF values for tips from FEM. 
            KIs_n:  
        
        """
        KIs_n = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_n, singularity_orders)
            * u_jump_n
        )
        slope_tip1, KI_tip1 = np.polyfit(r_n[0, :], KIs_n[0, :], 1)
        slope_tip2, KI_tip2 = np.polyfit(r_n[1, :], KIs_n[1, :], 1)
        KIs = np.array([KI_tip1, KI_tip2], dtype=float)

        return KIs, KIs_n