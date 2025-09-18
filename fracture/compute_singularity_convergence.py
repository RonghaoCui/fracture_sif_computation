
from __future__ import annotations

import porepy as pp
import numpy as np

import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from .fracture_displacement_jump import FractureDisplacementJump

class ComputeSingularityErrorForModel(FractureDisplacementJump):
    """Estimate near-tip singularity order (exponent) using single-scale or two-scale methods."""
    def __init__(self, model, exact_singularity_order: float = 0.5):
        super().__init__(model)
        self.exact_singularity_order = float(exact_singularity_order)

    def single_scale_option_normal(self, steps_pair):
        self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[0])}
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
        self.params["sampling"] = {"mode": "fixed", "steps": int(steps_pair[1])}
        u_jump_n_two, r_n_two = self.extract_sampling_normal_jump()

        print("Displacement jump n value 1: ", u_jump_n_one)
        print("Displacement jump n value 2: ", u_jump_n_two)
        print("Displacement jump n r 1: ", r_n_one)
        print("Displacement jump n r 2: ", r_n_two)
        
        return self.calculate_singularity_convergence(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two)
    
    def two_scale_option_normal(self, model_fine):
        u_jump_n_one, r_n_one = self.extract_sampling_normal_jump()
        u_jump_n_two, r_n_two = model_fine.extract_sampling_normal_jump()

        print("Displacement jump n value 1: ", u_jump_n_one)
        print("Displacement jump n value 2: ", u_jump_n_two)
        print("Displacement jump n r 1: ", r_n_one)
        print("Displacement jump n r 2: ", r_n_two)
        
        return self.calculate_singularity_convergence(u_jump_n_one, r_n_one, u_jump_n_two, r_n_two)
    
    def calculate_singularity_convergence(self, u_jump_n_one, r_n_one, u_jump_n_two, r_n_two): 
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