
from __future__ import annotations

import numpy as np

class ComputeSifValue:
    """
    Utilities for calculating stress intensity factor (SIF) values from displacement jumps 
    and distances to the fracture tip, as well as computing the exact SIF values for Sneddon's 
    and Westergaard's problems.
    
    The implementation supports: 
        - Single-point SIF estimation for KI and KII.
        - Two-point SIF estimation for KI and KII.
        - Multi-point SIF estimation for KI.
        - Two-scale SIF estimation for KI.

    Attributes: 
        params (dict): the model's parameters dictionary.
        solid (object): the solid material constants from the model parameters.
        mu (float):shear modulus.
        lam (float): Lame parameter.
        poi (float): Poisson's ratio.
        E (float): Young's modulus.
        frac_half_length (float): half length of the fracture.
    """
    def __init__(
            self, params: dict
    ) -> None:
        """
        Initialize parameters from SIF computation.

        Parameters:
            params (dict): A dictionary containing the model parameters.
        """
        self.params = params
        solid = self.params["material_constants"]["solid"]

        self.mu = solid.shear_modulus      # μ
        self.lam = solid.lame_lambda       # λ

        self.poi = self.lam / 2.0 / (self.mu + self.lam)      
        self.E = 2.0 * self.mu * (1.0 + self.poi)  

        self.frac_half_length = self.params["a"]

    
    def KI_from_one_point(
            self, 
            u_jump_n_one: np.ndarray, 
            r_n_one: np.ndarray, 
            singularity_orders: np.ndarray
    ) -> np.ndarray: 
        """
        Compute KI from given displacement jumps, distances to the fracture tip, 
        and singularity orders at a single point. 

        Parameters:
            u_jump_n_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_n_one (np.ndarray): Distances to tip for the first set.
            singularity_orders (np.ndarray): singularity orders for two tips.

        Returns:
            KIs: KI value array for two tips.

        """
        KIs = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_n_one, singularity_orders)
            * u_jump_n_one
        )

        return KIs
    
    def KI_from_two_points(
            self, 
            u_jump_n_one: np.ndarray, 
            r_n_one: np.ndarray, 
            u_jump_n_two: np.ndarray, 
            r_n_two: np.ndarray,
            singularity_orders: np.ndarray
    ) -> np.ndarray: 
        """
        Compute KI from given displacement jumps, distances to the fracture tip, 
        and singularity orders with two given points. 

        Parameters:
            u_jump_n_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_n_one (np.ndarray): Distances to tip for the first set.
            u_jump_n_two (np.ndarray): Second set (closer) of normal displacement jumps.
            r_n_two (np.ndarray): Distances to tip for the second set.
            singularity_orders (np.ndarray): singularity exponents for two tips

        Returns:
            KIs: KI value array for two tips.

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
    
    def KII_from_two_points(
            self, 
            u_jump_t_one: np.ndarray, 
            r_t_one: np.ndarray, 
            u_jump_t_two: np.ndarray, 
            r_t_two: np.ndarray,
            singularity_orders: np.ndarray
    ) -> np.ndarray: 
        """
        Compute KII from given displacement jumps, distances to the fracture tip, 
        and singularity orders with two given points. 

        Parameters:
            u_jump_t_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_t_one (np.ndarray): Distances to tip for the first set.
            u_jump_t_two (np.ndarray): Second set (closer) of normal displacement jumps.
            r_t_two (np.ndarray): Distances to tip for the second set.
            singularity_orders (np.ndarray): singularity exponents for two tips

        Returns:
            KIIs: KII value array for two tips.

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
    
    def KI_from_multipoints(
            self, 
            u_jump_n: np.ndarray, 
            r_n: np.ndarray, 
            singularity_orders: np.ndarray
    ) -> np.ndarray: 
        """
        Compute KI from given displacement jumps, distances to the fracture tip, 
        and singularity orders with given multiple points. 

        Parameters:
            u_jump_n (np.ndarray): A chosen set of normal displacement jumps.
            r_n (np.ndarray): Distances to tip for the chosen set.
            singularity_orders (np.ndarray): singularity exponents for two tips, i.e., 0.5.

        Returns:
            KIs: KI value array for two tips.

        """
        KIs_n = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_n, singularity_orders)
            * u_jump_n
        )
        num_tips = 2
        KIs = np.full(num_tips, np.nan)
        for i in [0, 1]:
            _, KIs[i] = self._fitted_SIF(KIs_n[:, i], r_n[:, i])

        return KIs
    
    def KI_from_two_scales(
        self, 
        u_jump_coarse:np.ndarray,
        u_jump_fine:np.ndarray,
        cell_size_coarse: float,
        refinement_level: int
    ) -> np.ndarray: 
        """
        Compute KI from coarse and fine models. 

        Parameters:
            u_jump_coarse (np.ndarray): normal displacement jumps from the coarse model.
            u_jump_fine (np.ndarray): normal displacement jumps from the fine model.
            cell_size_coarse (float): cell size of the coarse model.
            refinement_level (int): refinement level between the coarse and fine models.

        Returns:
            KIs: KI value array for two tips.
        
        Notes:
            This method has not been tested yet. It is based on the two-scale SIF estimation 
            method. 
        """
        
        coeff1 = (4.0*refinement_level + 5.0) / (4.0*refinement_level - 3.0)
        coeff2 = (2.0 / (4.0*refinement_level + 1.0) / cell_size_coarse) ** 0.5
        coeff3 = ((8.0*refinement_level + 2.0) / (4.0*refinement_level + 5.0)) ** (3.0/2.0)
        KIs = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.sqrt(2.0 * np.pi) * coeff1 * coeff2
            * (u_jump_fine * coeff3 - u_jump_coarse)
        )

        return KIs
    
    def calculate_exact_SIF_KI_sneddon(self) -> float: 
        """
        Calculate the analytical solution of SIF KI for Sneddon's problem
        in the opening mode. 

        Returns: 
            KI_exact: an analytical KI value.

        """
        pres0 = self.params["p0"]
        KI_exact =  pres0 * np.sqrt(np.pi * self.frac_half_length)

        return KI_exact

    def calculate_exact_SIF_KI_westergaard(self) -> float: 
        """
        Calculate the analytical solution of SIF KI for Westergaard's problem
        in the opening mode. 

        Returns: 
            KI_exact: an analytical KI value.

        """
        sigma_inf0 = self.params["sigma_inf"]
        KI_exact =  sigma_inf0 * np.sqrt(np.pi * self.frac_half_length)

        return KI_exact

    def calculate_exact_SIF_KII_westergaard(self) -> float: 
        """
        Calculate the analytical solution of SIF KII for Westergaard's problem
        in the shear mode. 

        Returns: 
            KII_exact: an analytical KII value.

        """
        tau_inf0 = self.params["tau_inf"]
        KII_exact =  tau_inf0 * np.sqrt(np.pi * self.frac_half_length)

        return KII_exact

    def _fitted_SIF(
        self, 
        SIFs_per_tip, 
        r_per_tip
    ) -> tuple[float, float]:
        """
        Calculate the linearly fitted SIF based on the multiple points. 

        Returns: 
            slopes: the fitted slope for the linear function.
            intercepts: the fitted intercept for the linear function, i.e., the final SIF.

        """
        slopes, intercepts = np.polyfit(r_per_tip, SIFs_per_tip, 1)

        return slopes, intercepts