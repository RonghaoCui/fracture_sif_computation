from __future__ import annotations

import porepy as pp
import numpy as np

class FEMComputeSifErrorConvergence:
    """
    FEM analogue of MPSA's SIF error convergence analysis in PorePy.
    """

    def __init__(
        self, 
        model: object | None = None,
        cache: dict | None = None,
    ) -> None:
        
        if model is not None:
            self._initialize_from_model(model)
        elif cache is not None:
            self._initialize_from_cache(cache)
        else:
            raise ValueError("Either a model or a cache must be provided.")
    
    def _initialize_from_model(self, model: object) -> None:
        self.model = model
        self.params = model.params

        solid = self.params["material_constants"]["solid"]
        self.mu = solid.shear_modulus          # μ
        self.lam = solid.lame_lambda           # λ
        self.poi = self.lam / 2.0 / (self.mu + self.lam)    # possion ratio
        self.E = 2.0 * self.mu * (1.0 + self.poi)           # Young's modulus

        # The tangential vector here is obtained by theta_rad, instead of coordinates from two tips.
        # We assume that they should be the same. 
        self.theta_rad = self.params["theta_rad"]        
        t = np.array([np.sin(self.theta_rad), np.cos(self.theta_rad)])

        self.tip_face_centers = model.tip_face_centers
        s_tip = self.tip_face_centers @ t
        tip0 = self.tip_face_centers[np.argmin(s_tip)]

        # Convert coordinates of tip face centers to the local system along the fracture 
        # as 1D coordinates, with tip0 as the origin. 
        # The coordinates of Tip 1 and Tip 2 are 0.0 and 2a, respectively.
        self.tip_face_centers = (self.tip_face_centers - tip0)@t
        # print("tip_face_centers: ", self.tip_face_centers)
        
        # Make sure tip_face_centers are in ascending order. 
        self.tip_face_centers = np.sort(self.tip_face_centers)

        (
            self.fracture_node_coords, 
            self.u_jump_normal, 
            self.u_jump_tangential
        ) = self.model.extract_displacement_jump()
        self.num_frac_nodes = self.fracture_node_coords.shape[0]
    
    def _initialize_from_cache(self, cache: dict) -> None:
        self.params = cache["params"]

        solid = self.params["material_constants"]["solid"]
        self.mu = solid.shear_modulus          # μ
        self.lam = solid.lame_lambda           # λ
        self.poi = self.lam / 2.0 / (self.mu + self.lam)    # possion ratio
        self.E = 2.0 * self.mu * (1.0 + self.poi)           # Young's modulus

        # The following parameters are directly from the cache, which should be processed
        # in above model initialization by considering the local system along the fracture.
        self.tip_face_centers = cache["tip_face_centers"]
        self.fracture_node_coords = cache["fracture_node_coords"]
        self.u_jump_normal = cache["u_jump_normal"]
        self.u_jump_tangential = cache["u_jump_tangential"]
        self.num_frac_nodes = self.fracture_node_coords.shape[0]

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
        x_coords = self.fracture_node_coords   
        current_frac_nodes = self.FEM_sampling_frac_cell(n_steps)
        # only [x,] and shape = (n_tips,)
        r_cell_to_tip = np.abs(self.tip_face_centers - x_coords[current_frac_nodes])
        u_jump = self.u_jump_normal[current_frac_nodes]
        return u_jump, r_cell_to_tip
    
    def FEM_extract_sampling_tangential_jump(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the tangential component of the displacement jump and corresponding radial distance to the
        closer tip with given sampling locations for two symmetrical tips. 

        Parameters: 
            n_steps: the n-th cell near the tip. 

        Returns: 
            u_jump: shape = (n_tips, )
                the tangential component of the displacement jump on the fracture.
            r_cell_to_tip: shape = (n_tips, )
                the radial distance to the closer fracture tip. 

        """
        x_coords = self.fracture_node_coords   
        current_frac_nodes = self.FEM_sampling_frac_cell(n_steps)
        # only [x, y] and shape = (n_tips, 2)
        r_cell_to_tip = np.abs(self.tip_face_centers - x_coords[current_frac_nodes])
        u_jump = self.u_jump_tangential[current_frac_nodes]
        return u_jump, r_cell_to_tip
    
    def FEM_sampling_frac_cell(self, n_steps: int) -> np.ndarray:
        """
        Get sampling indices of fracture nodes for both tips of one fracture. 

        Parameters: 
            n_steps: the n-th cell near the tip. 

        Returns: 
            current_frac_cells: shape = (2, )
                sampling indices of fracture nodes for tips 1 and 2 of one fracture.

        Notes: 
            It should be noted that there is no fracture tip node on fracture coords.

        """
        cells_tip_one = n_steps
        cells_tip_two = self.num_frac_nodes - n_steps - 1  
        current_frac_nodes = np.array([cells_tip_one, cells_tip_two], dtype=int)
        
        return current_frac_nodes
    
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
    
    def FEM_two_points_single_scale_tangential(
        self, sampling_pair: list
    ) -> tuple[float, np.ndarray]:
        """
        Calculate SIF KII by using the two-point method at one single scale of radial distance.

        Parameters: 
            sampling_pair: two nodes on fracture surfaces selected for two-point methods. 

        Returns: 
            Relative errors of calculated SIFs by FEM for tips, compared with analytical SIF values. 

        """
        u_jump_t_one, r_t_one = self.FEM_extract_sampling_tangential_jump(sampling_pair[1])
        u_jump_t_two, r_t_two = self.FEM_extract_sampling_tangential_jump(sampling_pair[0])
        
        return self.FEM_TwoPoints_SIF_KII_errors(u_jump_t_one, r_t_one,u_jump_t_two, r_t_two)
    
    def FEM_TwoPoints_SIF_KI_errors(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray,
        u_jump_n_two: np.ndarray, 
        r_n_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Calculate relative errors of calculated SIFs by FEM for tips, 
        compared with analytical SIF values. 

        The opening mode involves Sneddon's problem and Westergaard's problem,
        depending on whether the far-field stress `sigma_inf` is applied or not. 
        Here, we assume that if `sigma_inf` is not None, the problem is Westergaard's 
        problem in the opening mode. Otherwise, it is Sneddon's problem. 

        Parameters:
            u_jump_n_one: displacement jump of node 1
            r_n_one: radial distance to the tip of node 1
            u_jump_n_two: displacement jump of node 2
            r_n_two: radial distance to the tip of node 2

        Notes: 
            node 1 here is the one which is farther from the fracture tips. 
        """
        singularity_orders = 0.5
        KIs = self.FEM_KI_from_two_points(
            u_jump_n_one, 
            r_n_one, 
            u_jump_n_two, 
            r_n_two, 
            singularity_orders
        )
        sigma_inf = self.params.get("sigma_inf", None)

        if sigma_inf is not None:
            KI_exact = self.FEM_calculate_exact_SIF_KI_westergaard()
        else:
            KI_exact = self.FEM_calculate_exact_SIF_KI_sneddon()

        return self.FEM_relative_L2_error(KIs, KI_exact)
    
    def FEM_TwoPoints_SIF_KII_errors(
        self, 
        u_jump_t_one: np.ndarray, 
        r_t_one: np.ndarray,
        u_jump_t_two: np.ndarray, 
        r_t_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Calculate relative errors of calculated SIFs by FEM for tips, compared with analytical SIF values. 

        Parameters:
            u_jump_t_one: displacement jump of node 1
            r_t_one: radial distance to the tip of node 1
            u_jump_t_two: displacement jump of node 2
            r_t_two: radial distance to the tip of node 2

        Notes: 
            node 1 here is the one which is farther from the fracture tips. 
        """
        singularity_orders = 0.5
        KIIs = self.FEM_KII_from_two_points(
            u_jump_t_one, 
            r_t_one, 
            u_jump_t_two, 
            r_t_two, 
            singularity_orders
        )
        KII_exact = self.FEM_calculate_exact_SIF_KII_westergaard()

        return self.FEM_relative_L2_error(KIIs, KII_exact)
    
    def FEM_KI_from_two_points(
            self, 
            u_jump_n_one: np.ndarray, 
            r_n_one: np.ndarray, 
            u_jump_n_two: np.ndarray, 
            r_n_two: np.ndarray,
            singularity_orders: float,
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
    
    def FEM_KII_from_two_points(
            self, 
            u_jump_t_one: np.ndarray, 
            r_t_one: np.ndarray, 
            u_jump_t_two: np.ndarray, 
            r_t_two: np.ndarray,
            singularity_orders: float,
    ) -> np.ndarray: 
        """
        Calculate SIFs by FEM for tips based on extracted displacement jump. 
        
        Parameters:
            u_jump_t_one: displacement jump of node 1.
            r_t_one: radial distance to the tip of node 1.
            u_jump_t_two: displacement jump of node 2.
            r_t_two: radial distance to the tip of node 2.
            singularity_orders: singularity exponent, 0.5 by default.

        Returns: 
            KIs: SIF values for tips from FEM.  
        """
        KIIs_t_one = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_t_one, singularity_orders)
            * u_jump_t_one
        )

        KIIs_t_two = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_t_two, singularity_orders)
            * u_jump_t_two
        )

        KIIs = KIIs_t_one + r_t_one / (r_t_two - r_t_one) * (KIIs_t_one - KIIs_t_two)

        return KIIs
    
    def FEM_calculate_exact_SIF_KI_sneddon(self) -> float: 
        """
        Calculate the analytical solution of SIF KI for Sneddon's problem.

        Returns: 
            KI_exact: an analytical KI value.

        """
        pres0 = self.params["p0"]
        frac_half_length = self.params["a"]
        KI_exact =  pres0 * np.sqrt(np.pi * frac_half_length)

        return KI_exact
    
    def FEM_calculate_exact_SIF_KI_westergaard(self) -> float: 
        """
        Calculate the analytical solution of SIF KI for Westergaard's problem
        in the opening mode.

        Returns: 
            KI_exact: an analytical KI value.

        """
        sigma_inf0 = self.params["sigma_inf"]
        frac_half_length = self.params["a"]
        KI_exact =  sigma_inf0 * np.sqrt(np.pi * frac_half_length)

        return KI_exact
    
    def FEM_calculate_exact_SIF_KII_westergaard(self) -> float: 
        """
        Calculate the analytical solution of SIF KII for Westergaard's problem
        in the shear mode.

        Returns: 
            KII_exact: an analytical KII value.

        """
        tau_inf = self.params["tau_inf"]
        frac_half_length = self.params["a"]
        KII_exact =  tau_inf * np.sqrt(np.pi * frac_half_length)

        return KII_exact
    
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
        x_coords = self.fracture_node_coords
        current_frac_nodes = self.FEM_sampling_multi_frac_cells(n_steps)
        x_sample = x_coords[current_frac_nodes]
        r_cell_to_tip = np.vstack([
            x_sample[0,:] - self.tip_face_centers[0],
            self.tip_face_centers[1] - x_sample[1,:]
        ])
        u_jump = self.u_jump_normal[current_frac_nodes]
        return u_jump, r_cell_to_tip
    
    def FEM_extract_multi_points_tangential(
        self, sampling_points: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the tangential component of the displacement jump and corresponding radial distance to the
        closer tip with given sampling locations for two symmetrical tips. 

        Parameters: 
            sampling_points: the sampled group of cells near the tip. 

        Returns: 
            u_jump: shape = (n_tips, n_steps)
                the tangential component of the displacement jump on the fracture.
            r_cell_to_tip: shape = (n_tips, n_steps)
                the radial distance to the closer fracture tip. 

        Notes:         
            self.tip_face_centers only has [x, y] and shape = (n_tips,)    

        """
        sampling_points = np.array(sampling_points, dtype=int)
        n_steps = sampling_points
        x_coords = self.fracture_node_coords
        current_frac_nodes = self.FEM_sampling_multi_frac_cells(n_steps)
        x_sample = x_coords[current_frac_nodes]
        r_cell_to_tip = np.vstack([
            x_sample[0,:] - self.tip_face_centers[0],
            self.tip_face_centers[1] - x_sample[1,:]
        ])
        u_jump = self.u_jump_tangential[current_frac_nodes]
        return u_jump, r_cell_to_tip
    
    def FEM_sampling_multi_frac_cells(
        self, n_steps: np.ndarray
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
        cells_tip_two = self.num_frac_nodes - n_steps - 1  
        current_frac_nodes = np.vstack([cells_tip_one, cells_tip_two])
        
        return current_frac_nodes
    
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
            singularity_orders: float,
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
    
    def FEM_KII_from_multi_points(
            self, 
            u_jump_t: np.ndarray, 
            r_t: np.ndarray, 
            singularity_orders: float,
    ) -> tuple[np.ndarray, np.ndarray]: 
        """
        Calculate SIFs by FEM for tips based on extracted displacement jump. 
        
        Parameters:
            u_jump_t: displacement jump of sampled nodes.
            r_t: radial distance to the tip of sampled nodes.
            singularity_orders: singularity exponent, 0.5 by default.

        Returns: 
            KIIs: SIF values for tips from FEM. 
            KIIs_n:  
        
        """
        KIIs_t = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_t, singularity_orders)
            * u_jump_t
        )
        slope_tip1, KII_tip1 = np.polyfit(r_t[0, :], KIIs_t[0, :], 1)
        slope_tip2, KII_tip2 = np.polyfit(r_t[1, :], KIIs_t[1, :], 1)
        KIIs = np.array([KII_tip1, KII_tip2], dtype=float)

        return KIIs, KIIs_t
    
    def generate_sampling_pairs(self):
        k_min = 0

        # because of quadratic nodes
        k_max = self.num_frac_nodes // 2 - 1    

        pairs_in_cellsizes = []
        
        for k in range(k_min, k_max+1, 1):
            pairs_in_cellsizes.append([k, k + 1])

        return pairs_in_cellsizes
    
    def sampling_pairs_from_power_function(
            self, 
            C: float, 
            alpha: float,
    ) -> list[int]:

        h = self.params["meshing_arguments"]["cell_size"]
        a = self.params["a"]

        # The power function
        r_from_tip = C * (a ** (1 - alpha)) * (h ** alpha)

        # convert distance to cell index
        idx = int(np.round(r_from_tip / h * 2.0) - 1)
        # ensure index >= 0
        # idx = max(idx, 0)

        return [idx, idx + 1]    