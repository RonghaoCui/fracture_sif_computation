
from __future__ import annotations

import porepy as pp
import numpy as np

import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from .fracture_displacement_jump import FractureDisplacementJump
from .compute_singularity_convergence import ComputeSingularityErrorForModel
from .compute_SIF_value import ComputeSifValue


class ComputeSifErrorConvergence(FractureDisplacementJump):
    """
    Estimate fracture-tip singularity order (exponent) and its convergence 
    using single-scale or two-scale methods.

    This class extends :class:`FractureDisplacementJump` to compute convergence
    of near-tip singularity exponents using both single-scale and two-scale
    approaches. It provides utilities to measure the deviation between the
    numerically computed exponent and the analytical reference value.

    Attributes: 
        exact_singularity_order (float): Reference singularity order 
        (default 0.5).
    """
    def __init__(
        self, model: object
    ) -> None:
        """
        Initialize the SIF convergence computation class.

        Args: 
        model : object
            A PorePy-based model that contains the mixed-dimensional grid (mdg),
            displacement results, projection operators, and a ``params`` dictionary
        """
        super().__init__(model)

    def single_point_option_normal(
        self, steps_pair: tuple[int, int]
    ) -> tuple[float, np.ndarray]:
        """
        The single-point option with a single-point sampling displacement location.

        """
        sampling_cfg = self.params.get("sampling", {})
        step_mode = sampling_cfg.get("mode", None)

        if step_mode is None:
            raise ValueError("Sampling mode should be defined: 'refinement' or 'fixed'. ") 

        # Note: only the single-scale option("mode": 'fixed') should be specified with a location
        # No need to check the two-scale option ("mode": "refinement") when extracting the displacement
        # The point far from the fracture tip is chosen by default in steps_pair   
        if step_mode == "fixed": 
            self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[1])}
        
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()

        print("Displacement jump n value 2: ", u_jump_n_one)
        print("Displacement jump n r 2: ", r_n_one)
        
        return self.SinglePoint_SIF_KI_errors(u_jump_n_one, r_n_one)
    
    # def two_scale_option_normal(
    #     self, model_fine: FractureDisplacementJump
    # ) -> tuple[float, np.ndarray]:
    #     """
    #     The two-scale option using coarse and fine models at the given locations.

    #     """
    #     u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
    #     u_jump_n_two, r_n_two = model_fine.extract_sampling_normal_jump()

    #     print("Displacement jump n value 1: ", u_jump_n_one)
    #     print("Displacement jump n value 2: ", u_jump_n_two)
    #     print("Displacement jump n r 1: ", r_n_one)
    #     print("Displacement jump n r 2: ", r_n_two)
        
    #     return self.calculate_singularity_convergence(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two)
    
    def SinglePoint_SIF_KI_errors(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute SIF KI error estimates from one (u, r) sample and return errors.

        Args:
            u_jump_n_one (np.ndarray): First set of normal displacement jumps.
            r_n_one (np.ndarray): Distances to tip for the first set.
            u_jump_n_two (np.ndarray): Second set of normal displacement jumps.
            r_n_two (np.ndarray): Distances to tip for the second set.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.
        """
        SIFValue = ComputeSifValue(self.model)
        KIs = SIFValue.KI_from_exact_singularity(u_jump_n_one, r_n_one)
        print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        # Relative L2 error
        rel_l2_error_ave = (
            np.linalg.norm(KIs - KI_exact, ord=2) /
            (KI_exact * np.sqrt(len(KIs)))
        )
        # print("KI L2 error:", rel_l2_error_ave)

        rel_error_per_tip = np.abs(KIs - KI_exact) / KI_exact

        return rel_l2_error_ave, rel_error_per_tip