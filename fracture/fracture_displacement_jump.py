# Fracture tip singularity convergence analysis
# Only valid for one single fracture
from __future__ import annotations

import porepy as pp
import numpy as np

class FractureDisplacementJump: 
    # This class only can be done after having a mdg provided by the Model class (ManuSneddonModel2d here)
    def __init__(self, model):
        self.model = model
        self.mdg = model.mdg
        self.equation_system = model.equation_system

        self.sd_frac = self.mdg.subdomains(dim=1)[0]
        self.tip_faces = self.sd_frac.tags["tip_faces"].nonzero()[0]                 
        _, self.tip_cells = self.sd_frac.signs_and_cells_of_boundary_faces(self.tip_faces)  
        self.tip_face_centers = self.sd_frac.face_centers[:, self.tip_faces] 

        self.params = model.params
    
    def normal_displacement_jump(self):
        sd_frac = self.sd_frac
        u_jump = self.model.displacement_jump([sd_frac])                     
        normal_op = self.model.normal_component([sd_frac]) 
        u_jump_whole_frac = (normal_op @ u_jump).value(self.equation_system)
        return u_jump_whole_frac
    
    def tangential_displacement_jump(self):
        sd_frac = self.sd_frac
        u_jump = self.model.displacement_jump([sd_frac])                     
        tangential_op = self.model.tangential_component([sd_frac]) 
        u_jump_whole_frac = (tangential_op @ u_jump).value(self.equation_system)
        return u_jump_whole_frac

    def sampling_frac_cell(self):
        sampling_cfg = self.params.get("sampling", {})
        step_mode = sampling_cfg.get("mode", None)

        if step_mode is None:
            raise ValueError("Sampling mode should be defined: 'refinement' or 'fixed'. ")    

        if step_mode == "refinement":
            refinement_level = sampling_cfg.get("level", None)
            if refinement_level is None:
                raise ValueError("mode='refinement': keywords 'level' should be defined.")
            n_steps = 2 * refinement_level
        elif step_mode == "fixed": 
            n_steps_fixed = sampling_cfg.get("steps", None)
            if n_steps_fixed is None:
                raise ValueError("mode='fixed': 'steps' should be defined. ")
            n_steps = n_steps_fixed
        else: 
            raise ValueError(f"Unknown step_mode='{step_mode}'")
 
        current_frac_cells = np.where(self.tip_cells == 0, n_steps, self.tip_cells - n_steps)
        return current_frac_cells
    
    def extract_sampling_normal_jump(self):
        sd_frac = self.sd_frac
        current_frac_cells = self.sampling_frac_cell()
        r_cell_to_tip = np.linalg.norm(
            self.tip_face_centers - sd_frac.cell_centers[:, current_frac_cells], axis=0
        )
        u_jump = self.normal_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    def extract_sampling_tangential_jump(self):
        sd_frac = self.sd_frac
        current_frac_cells = self.sampling_frac_cell()
        r_cell_to_tip = np.linalg.norm(
            self.tip_face_centers - sd_frac.cell_centers[:, current_frac_cells], axis=0
        )
        u_jump = self.tangential_displacement_jump()[current_frac_cells]
        return u_jump, r_cell_to_tip
    
    