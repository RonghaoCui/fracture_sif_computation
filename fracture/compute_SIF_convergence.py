
from __future__ import annotations

from typing import Optional

import porepy as pp
import numpy as np

import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from .fracture_displacement_jump import FractureDisplacementJump
from .compute_singularity_convergence import ComputeSingularityErrorForModel
from .compute_SIF_value import ComputeSifValue


class ComputeSifErrorConvergence(ComputeSingularityErrorForModel):
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
        self.singularity_order: Optional[float] = None

    def single_point_single_scale_normal(
        self, steps_pair: tuple[int, int]
    ) -> tuple[float, np.ndarray]:
        """
        The single-point option with a single-scale sampling displacement location.

        """
        sampling_cfg = self.params.get("sampling", {})
        step_mode = sampling_cfg.get("mode", None) 

        # Note: only the single-scale option("mode": 'fixed') is used here
        if step_mode == "fixed": 
            self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[1])}
            u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
            self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[0])}
            u_jump_n_two, r_n_two = self.extract_sampling_normal_jump()
        else: 
            raise ValueError("Sampling mode 'fixed' should be required. ")

        print("Displacement jump n value 2: ", u_jump_n_one)
        print("Displacement jump n r 2: ", r_n_one)
        
        return self.SinglePoint_SIF_KI_errors(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
    
    def single_point_two_scales_normal(
        self, jumpmodel_fine: FractureDisplacementJump
    ) -> tuple[float, np.ndarray]:
        """
        The single-point option with a two-scale sampling displacement location.

        """
        sampling_cfg = self.params.get("sampling", {})
        step_mode = sampling_cfg.get("mode", None) 

        # Note: only the single-scale option("mode": 'fixed') is used here
        if step_mode == "refinement": 
            u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
            u_jump_n_two, r_n_two = jumpmodel_fine.extract_sampling_normal_jump()
        else: 
            raise ValueError("Sampling mode 'refinement' should be required. ")

        print("Displacement jump n value 1: ", u_jump_n_one)
        print("Displacement jump n value 2: ", u_jump_n_two)
        print("Displacement jump n r 1: ", r_n_one)
        print("Displacement jump n r 2: ", r_n_two)
        
        return self.SinglePoint_SIF_KI_errors(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
    
    def SinglePoint_SIF_KI_errors(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray,
        u_jump_n_two: np.ndarray, 
        r_n_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute SIF KI error estimates from two (u, r) samples and return errors.

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
        if self.singularity_order is None:
            singularity_orders = self.estimated_singularity(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
        else: 
            singularity_orders = self.singularity_order

        KIs = SIFValue.KI_from_one_point(u_jump_n_one, r_n_one, singularity_orders)
        print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        # Relative L2 error
        rel_l2_error_ave = (
            np.linalg.norm(KIs - KI_exact, ord=2) /
            (KI_exact * np.sqrt(len(KIs)))
        )

        rel_error_per_tip = np.abs(KIs - KI_exact) / KI_exact

        return rel_l2_error_ave, rel_error_per_tip

    def two_points_two_scales_normal(
        self, jumpmodel_fine: FractureDisplacementJump
    ) -> tuple[float, np.ndarray]:
        """
        The single-point option with a two-scale sampling displacement location.

        """
        sampling_cfg = self.params.get("sampling", {})
        step_mode = sampling_cfg.get("mode", None) 

        # Note: only the two-scale option("mode": 'refinement') is used here
        if step_mode == "refinement": 
            u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
            u_jump_n_two, r_n_two = jumpmodel_fine.extract_sampling_normal_jump()
        else: 
            raise ValueError("Sampling mode 'refinement' should be required. ")

        print("Displacement jump n value 1: ", u_jump_n_one)
        print("Displacement jump n value 2: ", u_jump_n_two)
        print("Displacement jump n r 1: ", r_n_one)
        print("Displacement jump n r 2: ", r_n_two)

        SIFValue = ComputeSifValue(self.model)
        if self.singularity_order == 0.5:
            KIs = SIFValue.KI_from_two_points(jumpmodel_fine)
        else:
            raise ValueError(f"Unsupported singularity_order={self.singularity_order}.")
        
        print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        # Relative L2 error
        rel_l2_error_ave = (
            np.linalg.norm(KIs - KI_exact, ord=2) /
            (KI_exact * np.sqrt(len(KIs)))
        )

        rel_error_per_tip = np.abs(KIs - KI_exact) / KI_exact

        return rel_l2_error_ave, rel_error_per_tip

