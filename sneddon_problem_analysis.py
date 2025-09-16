"""
Evaluate the singularity order (exponent) and stress intensity factor (SIF) 
at fracture tips within a simulation domain.

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
            Option 1: 

 - Perform both single-scale and two-scale convergence analysis. 
   The single-scale analysis evaluates convergence only with respect to mesh size (cell size: h),
   while the two-scale analysis involves coarse and fine models.


        

   
"""

import porepy as pp
import numpy as np

print("PorePy version:", pp.__version__)