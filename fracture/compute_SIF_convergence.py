
from __future__ import annotations

from typing import Optional

import numpy as np

from .fracture_displacement_jump import FractureDisplacementJump
from .compute_singularity_convergence import ComputeSingularityErrorForModel
from .compute_SIF_value import ComputeSifValue


class ComputeSifErrorConvergence(ComputeSingularityErrorForModel):
    """
    Estimate fracture-tip stress intensity factor (SIF) and its convergence 
    using single-scale or two-scale methods.
    
    The current script is only used for the opening mode (SIF KI).

    This class extends :class:`FractureDisplacementJump` to compute convergence
    of near-tip SIF using both single-scale and two-scale
    approaches. It provides utilities to measure the deviation between the
    numerically computed SIF and the analytical reference value.

    Attributes: 
        singularity_order (float): values manually provided by :class: `ComputeSifValue`
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
        self, sampling_pair: list
    ) -> tuple[float, np.ndarray]:
        """
        The single-point option with a single-scale sampling fracture cell.

        Args:
            steps_pair (tuple[int, int]): 
            Two integers ``(k0, k1)`` specifying the
                number of fracture cells away from the tip to sampling locations.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips .
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that steps_pair[1] is farther from the fracture tip 
            than steps_pair[0]

        """
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump(sampling_pair[1])
        u_jump_n_two, r_n_two = self.extract_sampling_normal_jump(sampling_pair[0])

        print("Displacement jump n value 2: ", u_jump_n_one)
        print("Displacement jump n r 2: ", r_n_one)
        
        return self.SinglePoint_SIF_KI_errors(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
    
    def two_points_single_scale_normal(
        self, sampling_pair: list
    ) -> tuple[float, np.ndarray]:
        """
        The two-point option for normal displacement solutions with a single-scale 
        sampling fracture cell.

        Parameters:
            steps_pair: shape = (tuple[int, int])
                Two integers specifying the number of fracture cells away from the tip 
                to sampling locations.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips .
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that steps_pair[1] is farther from the fracture tip 
            than steps_pair[0]

        """
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump(sampling_pair[1])
        u_jump_n_two, r_n_two = self.extract_sampling_normal_jump(sampling_pair[0])

        # print("Displacement jump n value 1: ", u_jump_n_one)
        # print("Displacement jump n value 2: ", u_jump_n_two)
        # print("Displacement jump n r 1: ", r_n_one)
        # print("Displacement jump n r 2: ", r_n_two)
        
        return self.TwoPoints_SIF_KI_errors(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
    
    def two_points_single_scale_tangential(
        self, sampling_pair: list
    ) -> tuple[float, np.ndarray]:
        """
        The two-point option for tangential displacement solutions with a single-scale 
        sampling fracture cell.

        Parameters:
            steps_pair: shape = (tuple[int, int])
                Two integers specifying the number of fracture cells away from the 
                tip to sampling locations.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips .
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that steps_pair[1] is farther from the fracture tip 
            than steps_pair[0]

        """
        u_jump_t_one, r_t_one = self.extract_sampling_tangential_jump(sampling_pair[1])
        u_jump_t_two, r_t_two = self.extract_sampling_tangential_jump(sampling_pair[0])

        # print("Displacement jump t value 1: ", u_jump_t_one)
        # print("Displacement jump t value 2: ", u_jump_t_two)
        # print("Displacement jump t r 1: ", r_t_one)
        # print("Displacement jump t r 2: ", r_t_two)
        
        return self.TwoPoints_SIF_KII_errors(u_jump_t_one, r_t_one,u_jump_t_two, r_t_two)
    
    def multipoints_single_scale_normal(
        self, sampled_cell: int, num_more_cells: int
    ) -> tuple[float, np.ndarray]:
        """
        The multipoint option with a single-scale sampling fracture cell.

        Args:
            steps_pair (tuple[int, int]): 
            Two integers ``(k0, k1)`` specifying the
                number of fracture cells away from the tip to sampling locations.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips .
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that steps_pair[1] is farther from the fracture tip 
            than steps_pair[0]

        """
        sampled_cells = np.arange(sampled_cell - num_more_cells, sampled_cell + num_more_cells + 1)
        # sampled_cells = np.arange(sampled_cell, sampled_cell + num_more_cells + 1)
        # sampled_cells = sampled_cells[(sampled_cells <= i_mid) & (sampled_cells >= k_min)]
        num_tips = 2
        u_jump_n = np.full((len(sampled_cells), num_tips), np.nan)
        r_n = np.full((len(sampled_cells), num_tips), np.nan)
        for i, cell_id in enumerate(sampled_cells): 
            u_jump_value, r_n_value = self.extract_sampling_normal_jump(cell_id)
            u_jump_n[i, :] = u_jump_value
            r_n[i, :] = r_n_value

        print("Displacement jump n value at the given sampled cell: ", u_jump_n[0, :])
        print("Displacement jump n at the given sampled cell: ", r_n[0, :])
        
        return self.multipoints_SIF_KI_errors(u_jump_n, r_n)
    
    def single_point_two_scales_normal(
        self, jumpmodel_fine: FractureDisplacementJump
    ) -> tuple[float, np.ndarray]:
        """
        The single-point option with a two-scale sampling fracture cell.
        
        Extract normal displacement jump samples and distances from
        ``self`` (coarse model) and from ``jumpmodel_fine`` (fine model) via 
        :meth:`extract_sampling_normal_jump`, then compute the relative errors 
        of the SIF using coarse models.

        Args:
            jumpmodel_fine (FractureDisplacementJump): Displacement jump extraction
                on the fine-mesh model. It must provide
                :meth:`extract_sampling_normal_jump`.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that ``params["sampling"]`` uses the 
            {"mode": "refinement", "level": refinement_level} option.

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
        
        return self.SinglePoint_SIF_KI_errors(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)

    def two_points_two_scales_normal(
        self, jumpmodel_fine: FractureDisplacementJump
    ) -> tuple[float, np.ndarray]:
        """
        The two-point option with two-scale sampling fracture cells.
        
        Extract normal displacement jump samples and distances from
        ``self`` (coarse model) and from ``jumpmodel_fine`` (fine model) via 
        :meth:`extract_sampling_normal_jump`, then compute the relative errors of the SIF
        using coarse and fine models.

        Args:
            jumpmodel_fine (FractureDisplacementJump): Displacement jump extraction
                on the fine-mesh model. It must provide
                :meth:`extract_sampling_normal_jump`.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that ``params["sampling"]`` uses the 
            {"mode": "refinement", "level": refinement_level} option.
            ``self.singularity_order`` must be set as 0.5. 
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

        SIFValue = ComputeSifValue(self.params)
        if self.singularity_order == 0.5:
            KIs = SIFValue.KI_from_two_scales(jumpmodel_fine)
        else:
            raise ValueError(f"Unsupported singularity_order={self.singularity_order}.")
        
        print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        return self._relative_L2_error(KIs, KI_exact)
    
    def SinglePoint_SIF_KI_errors(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray,
        u_jump_n_two: np.ndarray, 
        r_n_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute SIF KI error estimates from a single (u, r) sample and return errors.

        Args:
            u_jump_n_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_n_one (np.ndarray): Distances to tip for the first set.
            u_jump_n_two (np.ndarray): Second set (closer) of normal displacement jumps.
            r_n_two (np.ndarray): Distances to tip for the second set.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.
        """
        SIFValue = ComputeSifValue(self.params)
        if self.singularity_order is None:
            singularity_orders = self.estimated_singularity(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
        else: 
            singularity_orders = self.singularity_order

        KIs = SIFValue.KI_from_one_point(u_jump_n_one, r_n_one, singularity_orders)
        print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        return self._relative_L2_error(KIs, KI_exact)

    def TwoPoints_SIF_KI_errors(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray,
        u_jump_n_two: np.ndarray, 
        r_n_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute SIF KI error estimates from two (u, r) samples and return errors.
        Two singularity exponents can be used: exact value 0.5 and numerical singularity 
        exponents calculated from these two samples. 

        Parameters:
            u_jump_n_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_n_one (np.ndarray): Distances to tip for the first set.
            u_jump_n_two (np.ndarray): Second set (closer) of normal displacement jumps.
            r_n_two (np.ndarray): Distances to tip for the second set.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.
        """
        SIFValue = ComputeSifValue(self.params)
        if self.singularity_order is None:
            singularity_orders = self.estimated_singularity(u_jump_n_one, r_n_one,u_jump_n_two, r_n_two)
        else: 
            singularity_orders = self.singularity_order

        KIs = SIFValue.KI_from_two_points(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two, singularity_orders)
        # print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        return self._relative_L2_error(KIs, KI_exact)
    
    def TwoPoints_SIF_KII_errors(
        self, 
        u_jump_t_one: np.ndarray, 
        r_t_one: np.ndarray,
        u_jump_t_two: np.ndarray, 
        r_t_two: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute SIF KII error estimates from two (u, r) samples and return errors.
        Two singularity exponents can be used: exact value 0.5 and numerical singularity 
        exponents calculated from these two samples. 

        Parameters:
            u_jump_t_one (np.ndarray): First set (farther) of normal displacement jumps.
            r_t_one (np.ndarray): Distances to tip for the first set.
            u_jump_t_two (np.ndarray): Second set (closer) of normal displacement jumps.
            r_t_two (np.ndarray): Distances to tip for the second set.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.
        """
        SIFValue = ComputeSifValue(self.params)
        if self.singularity_order is None:
            singularity_orders = self.estimated_singularity(u_jump_t_one, r_t_one,u_jump_t_two, r_t_two)
        else: 
            singularity_orders = self.singularity_order

        KIIs = SIFValue.KII_from_two_points(u_jump_t_one, r_t_one, u_jump_t_two, r_t_two, singularity_orders)
        # print("KII at tips:", KIIs)

        KII_exact =  SIFValue.calculate_exact_SIF_KII_westargaard()

        return self._relative_L2_error(KIIs, KII_exact)
    
    def multipoints_SIF_KI_errors(
        self, 
        u_jump_n: np.ndarray, 
        r_n: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute SIF KI error estimates from multiple (u, r) samples and return errors.
        Only the exact singularity exponent 0.5 is used. 

        Args:
            u_jump_n (np.ndarray): A chosen set of normal displacement jumps for two tips.
            r_n (np.ndarray): Distances to tip for the chosen set for two tips.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips.
                ``rel_error_per_tip``: Relative absolute error array from two tips.
        """
        SIFValue = ComputeSifValue(self.params)
        if self.singularity_order != 0.5:
            raise ValueError(f"The singularity exponent should be 0.5.")

        KIs = SIFValue.KI_from_multipoints(u_jump_n, r_n, self.singularity_order)
        print("KI at tips:", KIs)

        KI_exact =  SIFValue.calculate_exact_SIF_KI_sneddon()

        return self._relative_L2_error(KIs, KI_exact)

    def _relative_L2_error(
        self, SIFs: np.ndarray, SIF_exact: float
    ) -> tuple[float, np.ndarray]:
        """
        Compute relative errors of SIF against an exact reference.

        Args:
            SIFs (np.ndarray): Estimated SIF values per tip (shape e.g. `(2,)` for two tips).
            SIF_exact (float): Exact SIF used as the scalar reference.

        Returns:
            rel_l2_error_ave: Relative L2 error averaged by two tips.
            rel_error_per_tip: Relative absolute error array from two tips.
        """
        rel_l2_error_ave = (
            np.linalg.norm(SIFs - SIF_exact, ord=2) /
            (SIF_exact * np.sqrt(len(SIFs)))
        )

        rel_error_per_tip = np.abs(SIFs - SIF_exact) / SIF_exact

        return rel_l2_error_ave, rel_error_per_tip
    
