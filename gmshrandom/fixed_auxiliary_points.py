from __future__ import annotations

from gmshrandom.auxiliary_points_in_mesh import AuxiliaryPointsMixin

class FixedAuxiliaryPointsMixin(AuxiliaryPointsMixin):

    def _random_auxiliary_points(self) -> list[tuple[float, float]]:
        return self._fixed_auxiliary_points()
    
    def _fixed_auxiliary_points(self) -> list[tuple[float, float]]:

        L = self.params["domain_size"]
        # theta_rad = self.params["theta_rad"]
        cell_size = self.params["meshing_arguments"]["cell_size"]
        a = self.params["a"]
        
        # Central and horizontal fracture along x-axis, with two tips
        y0 = L / 2
        xL = L / 2 - a
        xR = L / 2 + a

        r = cell_size * 0.8

        pts = [
            # left tip
            (xL - r, y0 + 0.6 * r), 
            (xL - r, y0 - 0.6 * r),
            
            # right tip
            (xR + r, y0 + 0.6 * r), 
            (xR + r, y0 - 0.6 * r),
        ]

        # dx = cell_size * 0.866
        # dy = cell_size * 0.5

        # pts = [
        #     # left tip
        #     (xL - dx, y0 + dy), 
        #     (xL - dx, y0 - dy),
            
        #     # right tip
        #     (xR + dx, y0 + dy), 
        #     (xR + dx, y0 - dy),
        # ]

        # pts = [
        #     # left tip
        #     (xL - r, y0), 
            
        #     # right tip
        #     (xR + r, y0), 
        # ]

        return pts

