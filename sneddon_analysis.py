"""
Evaluate the singularity order (exponent) and stress intensity factor (SIF) 
at fracture tips within a simulation domain for Sneddon's problem.

Currently, the script only supports a single fracture with two fracture tips.

Features: 
 - Provide several options for selecting evaluation locations (cell centers) along the fracture. 
   Singularity order (exponent): 
        Single-scale option: evaluate singularity at the same grid refinement level by two cell centers (i and j)
        near the fracture tip. The cell center i is selected at the fracture tip cell (r_i=h/2). The cell center j can 
        be movable to the third cell closest to the frature tip (r_j=5*h/2). 

        Two-scale option: cell centers i and j, whose distance are r_i and r_j, are movable and selected from fine and 
        coarse grids, respectively. If the refinement levels of coarse and fine grids are k_c and k_f, where k_f=k_c+1, 
        the distance from cell center j to the fracture tip face in coarse grids is (4*k_c+1)*h_c/2 and the distance from 
        cell center j in fine grids is (4*k_f+1)*h_f/2. 
    
    Stress intensity factor (SIF) KI: 
        Single-scale option: 
            Option 1: The location is selected on the third cell closest to the frature tip (r_j=5*h/2). 
            The singularity order is set as 0.5.
            Option 2: The location is the same as Option 1, but the singularity order is from the single-scale option.
            Option 3: The location is selected on cell center j from the two-scale option. The singularity order is also 
            from the two-scale option. 

        Two-scale option: the expression is derived from theoretical solution with elimination of the second term. 

 - Perform both single-scale and two-scale convergence analysis. 
   The single-scale analysis evaluates convergence only with respect to mesh size (cell size: h),
   while the two-scale analysis involves coarse and fine models.

"""

import porepy as pp
import numpy as np

print("PorePy version:", pp.__version__)

import math
import scipy.sparse as sps

from tests.functional.setups.manu_sneddon_2d import ManuSneddonModel2d

a = 0.3
poi = 0.25
shear_modulus = 1.0
lam = 2 * shear_modulus * poi / (1 - 2 * poi)

params = {
    "prepare_simulation": True,
    "material_constants": {"solid": pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)},
    "a": a,
    "domain_size": 1.0,
    "p0": 1e-4,
    "poi": poi,
    "meshing_arguments": {"cell_size": 0.1},
    "grid_type": "simplex",
    "theta_rad": math.radians(90 - 0),
    "num_bem_segments": 1000,
    "error_exclusion_zone_fracture_tips": 0.1,
}

model = ManuSneddonModel2d(params=params)
pp.run_time_dependent_model(model)

# def get_u_a_and_u_b(model):
#     frac_sd = model.mdg.subdomains(dim=model.nd - 1)
#     nd_vec_to_normal = model.normal_component(frac_sd)
#     u_n = (nd_vec_to_normal @ model.displacement_jump(frac_sd)).value(
#             model.equation_system
#         )
#     u_n_a = u_n[2]
#     u_n_b = u_n[-3]
#     return u_n_a, u_n_b

# Plot
pp.plot_grid(
    model.mdg,
    rgb=[1, 1, 1],
    figsize=(8,8), 
    plot_2d=True,           
    linewidth=0.3,
    alpha=1.0,
    title="Grid before fracture propagation"
)

# 2D displacement plot
for g, data in model.mdg.subdomains(return_data=True):
    # get 'u'（t=0 by default）
    u = data.get(pp.TIME_STEP_SOLUTIONS, {}).get("u", {}).get(0, None)

    if u is not None:
        # get the displacement y along the y direction
        y_disp = u[1::2]

        # plot
        pp.plot_grid(
            g,
            cell_value=y_disp,
            plot_2d=True,
            title="Y-direction displacement"
        )
    else:
        print(f"Grid {g} has no 'u' defined in TIME_STEP_SOLUTIONS")

###################################################
# Fracture tip singularity convergence analysis
###################################################
class FractureDisplacementJump: 
    # This class only can be done after having a mdg provided by the Model class (ManuSneddonModel2d here)
    def __init__(self, model):
        self.model = model
        self.mdg = model.mdg
        self.equation_system = model.equation_system

    def _internal_faces_of_cell(self, cell_faces: sps.csr_matrix, c: int):
        """Return internal faces (shared by 2 cells) of cell c, excluding boundary faces."""
        faces = cell_faces[:, c].nonzero()[0]
        internal = []
        for f in faces:
            # number of cells on face f is 2 for internal faces
            # .indices returns non-zero index in the sparse matrix
            # cell_faces is a sparse matrix from .tocsr()
            if cell_faces[f, :].indices.size == 2:      
                internal.append(f)
        return internal

    def _other_cell_across_face(self, cell_faces: sps.csr_matrix, face_id: int, cell_id: int):
        """Given an internal face and one of its incident cells, return the other cell in 1D."""
        inc = cell_faces[face_id, :].indices
        # Check whether the face inc is internal or not
        if inc.size != 2:
            return None
        return inc[0] if inc[1] == cell_id else inc[1]

    # Determine the fracture cell which will be selected
    def _walk_inward_from_tip(self, k_idx: int, cell_faces: sps.csr_matrix, f_tip: int, c_tip: int, n_steps: int):
        current_cell = c_tip
        prev_face = f_tip
        steps = 0
        print(f"[Tip {k_idx}] Step {steps}: prev_face={prev_face}, current_cell={current_cell}")

        while steps < n_steps:
            cand_faces = self._internal_faces_of_cell(cell_faces, current_cell)
            next_face = None
            for f in cand_faces:
                if f != prev_face:
                    next_face = f
                    break
            if next_face is None:
                return -1
            next_cell = self._other_cell_across_face(cell_faces, next_face, current_cell)
            if next_cell is None:
                return -1
            prev_face = next_face
            current_cell = next_cell
            steps += 1
            print(f"[Tip {k_idx}] Step {steps}: prev_face={prev_face}, current_cell={current_cell}")

        return current_cell
    
    # Calculate displacement jump and distance to the tip face at a specified cell center
    def extract_displacement_jump(self, k_target: int):
        # 1: Get fracture cells
        sd_frac = self.mdg.subdomains(dim=1)[0]

        # Find the index of the existing tag "tip_faces"， .tags only returns 0 or 1
        tip_faces = sd_frac.tags["tip_faces"].nonzero()[0] 

        # Get cells in sd_secondary on the fracture tips (from conforming_propagation)
        _, tip_cells = sd_frac.signs_and_cells_of_boundary_faces(tip_faces)  

        # 2: Get AD operator based on displacement jump
        u_jump = self.model.displacement_jump([sd_frac])                     
        normal_op = self.model.normal_component([sd_frac])                   
        # Tangential component
        # tang_op = model.tangential_component([sd_frac])              

        # 3: Get values from AD operators
        # COD normal
        cod_whole_frac = (normal_op @ u_jump).value(self.equation_system)       # shape: (num_frac_cells,)
        # COD tangential 
        # cod_t = (tang_op @ u_jump).value(model.equation_system)       # shape: (num_frac_cells,)

        # Location of the tip face centers
        tip_face_centers = sd_frac.face_centers[:, tip_faces]           # (nd, n_tips)
        # cc = sd_frac.cell_centers[:, tip_cells]                       # (nd, n_tips)
        # r_tip = np.linalg.norm(fc - cc, axis=0)                       # (n_tips,)
        # print("The distance r: ", r_tip)

        # neighbor walking
        cell_faces = sd_frac.cell_faces.tocsr()

        # Get the third cell center from the tip: (r = (4k+1)h/2)
        n_steps = 2 * k_target
        neighbor_n_cells = np.full(tip_cells.shape, -1, dtype=int)
        r_n = np.full(tip_cells.shape, np.nan)
        cod_n = np.full(tip_cells.shape, np.nan)

        # Find the (2k+1)th and (2k-1)th cells inward from the tip cells: two fracture tips for 1D fracture here
        for k_idx, (f_tip, c_tip) in enumerate(zip(tip_faces, tip_cells)):
            # Fracture cell n
            current_cell = self._walk_inward_from_tip(k_idx, cell_faces, f_tip, c_tip, n_steps)
        
            if current_cell != -1:
                neighbor_n_cells[k_idx] = current_cell
                # Distance 
                r_n[k_idx] = np.linalg.norm(tip_face_centers[:, k_idx] - sd_frac.cell_centers[:, current_cell])
                # Normal displacemen jump at the current cell center
                cod_n[k_idx] = cod_whole_frac[current_cell]

        return cod_n, r_n, tip_face_centers

    
    # Need to be modified: Match the order of the fine fracture tips to that of the coarse mesh
    def match_tips_by_position(self, tip_face_coarse, tip_face_fine):
        """ Match the tip order from coarse and fine mesh by calculating the distance
            Return the correct order of the fine mesh matching the coarse mesh
        """
        n_tips = tip_face_coarse.shape[1]
        updated_order = np.empty(n_tips, dtype=int)
        used = set()
        for i in range(n_tips):
            diffs = np.linalg.norm(tip_face_fine.T - tip_face_coarse[:, i], axis=1)
        
            sorted_idx = np.argsort(diffs)
            for j in sorted_idx:
                if j not in used:
                    updated_order[i] = j
                    used.add(j)
                    break
        return updated_order


COD = FractureDisplacementJump(model)
# cod_n_coarse, r_n_coarse, tip_face_coarse = extract_displacement_jump(model_coarse, k_target_coarse)
cod_n, r_n, tip_face_centers = COD.extract_displacement_jump(k_target=1)

print("COD value: ", cod_n)
print("COD r: ", r_n)