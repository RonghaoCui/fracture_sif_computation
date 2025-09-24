
from __future__ import annotations

import porepy as pp
import numpy as np

import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from .fracture_displacement_jump import FractureDisplacementJump

class ComputeSingularityErrorForModel(FractureDisplacementJump):
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
        self, model: object, exact_singularity_order: float = 0.5
    ) -> None:
        """
        Initialize the singularity error computation class.

        Args: 
        model : object
            A PorePy-based model that contains the mixed-dimensional grid (mdg),
            displacement results, projection operators, and a ``params`` dictionary
        exact_singularity_order : float, optional
            Exact analytical singularity order as the reference value
            (default is 0.5).
        """
        super().__init__(model)
        self.exact_singularity_order = float(exact_singularity_order)

    def single_scale_option_normal(
        self, steps_pair: tuple[int, int]
    ) -> tuple[float, np.ndarray]:
        """
        The single-scale option on the same model with two sampling displacement locations.

        This method sets the sampling configuration twice (two different
        ``steps`` values under ``{"mode": "fixed"}``), extracts the normal
        displacement jump and corresponding distances to the tip, and then
        estimates the exponent by comparing the two samples.

        Args:
            steps_pair (tuple[int, int]): 
            Two integers ``(k0, k1)`` specifying the
                number of fracture cells away from the tip to sample at each pass.

        Returns:
            Tuple[float, np.ndarray]:
                ``rel_l2_error_ave``: Relative L2 error averaged by two tips .
                ``rel_error_per_tip``: Relative absolute error array from two tips.

        Notes:
            This method assumes that ``params["sampling"]`` uses the
              ``{"mode": "fixed", "steps": k}`` option.
        """
        self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[0])}
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
        self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[1])}
        u_jump_n_two, r_n_two = self.extract_sampling_normal_jump()

        print("Displacement jump n value 1: ", u_jump_n_one)
        print("Displacement jump n value 2: ", u_jump_n_two)
        print("Displacement jump n r 1: ", r_n_one)
        print("Displacement jump n r 2: ", r_n_two)
        
        return self.calculate_singularity_convergence(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two)
    
    def two_scale_option_normal(
        self, model_fine: FractureDisplacementJump
    ) -> tuple[float, np.ndarray]:
        """
        The two-scale option using coarse and fine models at the given locations.

        Extracts normal displacement jump samples and distances from
        ``self`` (coarse model) and from ``model_fine`` (fine model), then
        estimates the exponent by comparing the two samples.

        Args:
            model_fine (FractureDisplacementJump): Displacement jump extraction
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
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
        u_jump_n_two, r_n_two = model_fine.extract_sampling_normal_jump()

        print("Displacement jump n value 1: ", u_jump_n_one)
        print("Displacement jump n value 2: ", u_jump_n_two)
        print("Displacement jump n r 1: ", r_n_one)
        print("Displacement jump n r 2: ", r_n_two)
        
        return self.calculate_singularity_convergence(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two)
    
    def calculate_singularity_convergence(
        self, 
        u_jump_n_one: np.ndarray, 
        r_n_one: np.ndarray, 
        u_jump_n_two: np.ndarray, 
        r_n_two: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Compute exponent error estimates from two (u, r) samples and return errors.

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
        # Get log10
        log_u_jump_n_one = np.log10(np.abs(u_jump_n_one))
        log_u_jump_n_two   = np.log10(np.abs(u_jump_n_two))
        log_r_n_one   = np.log10(r_n_one)
        log_r_n_two   = np.log10(r_n_two)

        singularity_orders = (log_u_jump_n_one - log_u_jump_n_two) / (log_r_n_one - log_r_n_two)
        singularity_order_ave = np.mean(singularity_orders)

        print("The order of singularity:", singularity_orders.tolist())
        print("The order average of singularity:", singularity_order_ave)

        abs_l2_error_ave = np.linalg.norm(singularity_orders - self.exact_singularity_order, ord=2)

        # Relative L2 error
        rel_l2_error_ave = (
            abs_l2_error_ave /
            (self.exact_singularity_order * np.sqrt(len(singularity_orders)))
        )

        ####### error for each tips
        # Absolute error per tip
        abs_l2_error_tip = np.abs(singularity_orders - self.exact_singularity_order)
        # Relative error per tip
        rel_error_per_tip = abs_l2_error_tip / self.exact_singularity_order

        return rel_l2_error_ave, rel_error_per_tip
    
    def get_number_of_fracture_cells(self) -> int: 
        """
        Return the number of cells on the 1D fracture subdomain.

        This assumes a single fracture (the 1D subdomain in the model).

        Returns:
            int: Number of 1D fracture cells.
        """
        # Get 1D fracture grid (in 2D model)
        fracture_grids = self.mdg.subdomains(dim=1)
        # Usually only one fracture in Sneddon problem
        fracture_grid = fracture_grids[0]
        # Number of fracture cells
        num_frac_cells = fracture_grid.num_cells

        return num_frac_cells