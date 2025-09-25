
from __future__ import annotations

import porepy as pp
import numpy as np

import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from .fracture_displacement_jump import FractureDisplacementJump
from .compute_singularity_convergence import ComputeSingularityErrorForModel


class ComputeSifValue(FractureDisplacementJump):
    """
    Calculate methods for calculating stress intensity factor (SIF) values.

    Attributes: 
        mu (float):shear modulus
        lam (float): Lame parameter
        poi (float): Poisson's ratio
        E (float): Young's modulus
    """
    def __init__(
            self, model: object
    ) -> None:
        """
        Initialize elastic constants from the model.

        Args:
            model: A solved PorePy model instance
        """
        # Calculate SIF KI with given parameters in the model
        super().__init__(model)
        solid = self.params["material_constants"]["solid"]
        self.mu = solid.shear_modulus          # μ
        self.lam = solid.lame_lambda           # λ
        self.poi = self.lam / 2.0 / (self.mu + self.lam)      # possion ratio
        self.E = 2.0 * self.mu * (1.0 + self.poi)        # Young's modulus
    
    def KI_from_one_point(
            self, 
            u_jump_n_one: np.ndarray, 
            r_n_one: np.ndarray, 
            singularity_orders: np.ndarray
    ) -> np.ndarray: 
        """
        Compute KI from given displacement jumps, distances to the fracture tip, 
        and singularity orders at a single point. 

        Args:
            u_jump_n_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_n_one (np.ndarray): Distances to tip for the first set.
            singularity_orders (np.ndarray): singularity orders for two tips

        Returns:
            np.ndarray:
                ``KIs``: KI value array for two tips.

        """
        KIs = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.power(2.0 * np.pi / r_n_one, singularity_orders)
            * u_jump_n_one
        )

        return KIs
    
    def KI_from_two_points(
            self, jumpmodel_fine: FractureDisplacementJump
    ) -> np.ndarray: 
        """
        Compute KI from coarse and fine models. 

        Args:
            jumpmodel_fine (FractureDisplacementJump): Displacement jump extraction
            on the fine-mesh model. It must provide
            :meth:`extract_sampling_normal_jump`.

        Returns:
            np.ndarray:
                ``KIs``: KI value array for two tips.

        """
        sampling_cfg = self.params.get("sampling", {})
        step_mode = sampling_cfg.get("mode", None)

        if step_mode == "refinement":
            refinement_level = sampling_cfg.get("level", None)
        else:
            raise ValueError("Sampling mode should be defined: 'refinement'. ") 
        
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
        u_jump_n_two, r_n_two = jumpmodel_fine.extract_sampling_normal_jump()
        h = self.params["meshing_arguments"]["cell_size"]
        
        coeff1 = (4.0*refinement_level + 5.0) / (4.0*refinement_level - 3.0)
        coeff2 = (2.0 / (4.0*refinement_level + 1.0) / h) ** 0.5
        coeff3 = ((8.0*refinement_level + 2.0) / (4.0*refinement_level + 5.0)) ** (3.0/2.0)
        KIs = (
            self.E / (8.0 * (1 - self.poi**2.0))
            * np.sqrt(2.0 * np.pi) * coeff1 * coeff2
            * (u_jump_n_two * coeff3 - u_jump_n_one)
        )

        return KIs
    
    def calculate_exact_SIF_KI_sneddon(self) -> float: 
        """
        Calculate the analytical solution of SIF KI for Sneddon's problem

        Returns: 
            float:
                ``KI_exact``: an analytical KI value.

        """
        pres0 = self.params["p0"]
        frac_half_length = self.params["a"]
        KI_exact =  pres0 * np.sqrt(np.pi * frac_half_length)

        return KI_exact
