from __future__ import annotations
from pathlib import Path

import numpy as np
import porepy as pp

from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.gmsh_interface import GmshWriter
# from porepy.fracs.gmsh_interface import Tags as GmshInterfaceTags
import porepy.fracs.simplex
import gmsh
import random
import gc

class AuxiliaryPointsMixin:
    """
    A mixin for adding auxiliary points (nodes) into the mesh to make simplex grids 
    have some randomness in the 2D matrix. 

    This mixin overrides the geometry construction by implementing a custom method 
    for :meth:`create_mdg()` . 
    """
    def create_fracture_network(self) -> None:
        """Set the fracture network from the fractures and domain."""
        self.fracture_network_2d = FractureNetwork2d(self._fractures, self.domain, tol=1e-8)

    def create_mdg(self) -> None:
        """
        Get the mdg with auxiliary points. Necessary properties (pts, mesh_size) in GmshData 
        are modified for auxiliary points. 
        """
        frac_net = self.fracture_network_2d
        cell_size = self.params["meshing_arguments"]["cell_size"]

        # Using the same cell size because cells are expected to be uniform in the domain.
        mesh_args = {
            "mesh_size_frac": cell_size,
            "mesh_size_bound": cell_size,
            "mesh_size_min": cell_size,  
        }

        gmsh_repr = frac_net.prepare_for_gmsh(
            mesh_args=mesh_args,
            tol=frac_net.tol,
            do_snap=False,         
            constraints=None,
            dfn=False,
        )

        aux_pts = self._random_auxiliary_points()
        aux_pts = np.asarray(aux_pts, dtype=float).T  # return shape=(2, N)
        self._aux_points_xy = aux_pts.T               # return shape=(N, 2) only for plotting

        auxpts_cell_sizes = np.full(aux_pts.shape[1], fill_value=cell_size, dtype=float)

        gmsh_repr.pts = np.hstack([gmsh_repr.pts, aux_pts])                             # (2, N+Naux)
        gmsh_repr.mesh_size = np.concatenate([gmsh_repr.mesh_size, auxpts_cell_sizes])  # (N+Naux, )

        start_idx = gmsh_repr.pts.shape[1] - aux_pts.shape[1]
        # if gmsh_repr.physical_points is None:
        #     gmsh_repr.physical_points = {}
        # for i in range(aux_pts.shape[1]):
        #     gmsh_repr.physical_points[start_idx + i] = GmshInterfaceTags.FRACTURE_TIP
        writer = GmshWriter(gmsh_repr)
        print("\n Gmsh writing finished. ")

        added_point_tags = [
            writer._point_tags[idx] 
            for idx in range(
                start_idx, start_idx + aux_pts.shape[1]
            )
        ]

        gmsh.model.mesh.embed(0, added_point_tags, 2, writer._domain_tag)
        gmsh.model.geo.synchronize() 

        msh_name = Path("tmp_aux_points.msh")
        writer.generate(msh_name, ndim=2, write_geo=False, finalize=False, clear_gmsh=True)

        subdomains = porepy.fracs.simplex.triangle_grid_from_gmsh(msh_name, constraints=None) 

        del gmsh_repr
        del writer
        gc.collect()

        mdg = pp.meshing.subdomains_to_mdg(subdomains)
        
        del subdomains
        gc.collect()
        self.mdg = mdg
        print("\n mdg finished. ")


    def _random_auxiliary_points(self) -> list[tuple[float, float]]:
        """
        Generate auxiliary points in two predefined rectangular domains located above and 
        below the horizontal fracture. The total number of auxiliary points is given by 
        ``self.params["aux_point_count"]``. The domain for auxiliary points are defined with a 
        margin (whose length is the given cell size) along domain boundaries and fracture. 

        Returns: 
            pts: the list of coordinates (x,y) of auxiliary points.      
        """
        L = self.params["domain_size"]
        # theta_rad = self.params["theta_rad"]
        cell_size = self.params["meshing_arguments"]["cell_size"]
        n_tot = int(self.params["aux_point_count"])
        n_up = n_tot // 2
        n_lo = n_tot - n_up

        xmin, xmax = cell_size, L - cell_size
        y_up_min, y_up_max = L/2 + cell_size, L - cell_size
        y_lo_min, y_lo_max = cell_size, L/2 - cell_size

        up_rect = (xmin, xmax, y_up_min, y_up_max)
        lo_rect = (xmin, xmax, y_lo_min, y_lo_max)
    
        min_dist = self.params["aux_point_min_dist"]
        seed = self.params["aux_seed"]
        max_tries = 50
        rng = random.Random(seed)
        
        pts_up = self._rand_points_in_rect(n_up, up_rect, min_dist, max_tries, rng)
        pts_lo = self._rand_points_in_rect(n_lo, lo_rect, min_dist, max_tries, rng)
        
        pts = pts_up + pts_lo
        return pts
    
    def _rand_points_in_rect(
            self, 
            n: int, 
            rect: list[float, float, float, float], 
            min_dist: float, 
            max_tries: int, 
            rng: int
    )  -> list[tuple[float, float]]:
        """
        Random generation of auxiliary points in a rectangular domain with a constraint of 
        the distance between two points larger than ``min_dist``. The randomness is controlled 
        by a given random number generator. 

        Parameters: 
            n: the number of generated points.
            rect: rectangular domain shape.
            min_dist: the minimum distance.
            max_tries: the maximum tries of insert points.
            rng: the seed.

        Returns:
            pts: the list of coordinates (x,y) of points. 
        """
        xmin, xmax, ymin, ymax = rect
        cell = min_dist
        inv = 1.0 / cell

        grid = {}
        pts = []

        r2 = min_dist * min_dist
        tries = 0
        max_tries_tol = max_tries * n

        while len(pts) < n and tries < max_tries_tol:
            tries += 1
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            gi, gj = int((x - xmin) * inv), int((y - ymin) * inv)

            ok = True
            for ii in range(gi - 1, gi + 2):
                for jj in range(gj - 1, gj + 2):
                    for k in grid.get((ii, jj), []):
                        px, py = pts[k]
                        if (x - px)**2 + (y - py)**2 < r2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                idx = len(pts)
                pts.append((x, y))
                grid.setdefault((gi, gj), []).append(idx)
        
        print("The number of points: ", len(pts))
        return pts

